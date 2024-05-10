import json

import cv2
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import glob

from torchsummary import summary
from tqdm import tqdm
import os
import gc
from PIL import Image
from utils.dataset import GIANA
import utils.transforms as t
from models.unet import *
from models.transunet import *
from models.Deeplabv3_plus_mobilev2 import *
from models.SA_deepLabV3plus import *
from models.Deeplab_v3_plus import *
from utils.avg import AverageMeter
from utils.evaluation import Evaluation
from utils.visualize import Dashboad
from utils.losses import *
from utils.post_process import *
from settings import get_arguments
import sys
from thop import profile
import itertools
import time
import warnings
from utils.compute_sdf import compute_sdf
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.nn.utils.prune as prune
from DNN_printer import DNN_printer
warnings.filterwarnings('ignore')
# https://github.com/saeedizadi/binseg_pytoch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# 参考：https://blog.csdn.net/qq_43219379/article/details/124003959
def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def load_data(args):
    # compute_mean(args)
    # # old data: 800
    # normalize = t.Normalize(mean=[0.7203, 0.7054, 0.5127], std=[0.1517, 0.0494, 0.1247])

    # new data: 1200
    # 1600: [0.7479, 0.7204, 0.5454], std=[0.1594, 0.0573, 0.1386]
    normalize = t.Normalize(mean=[0.7249, 0.7155, 0.5255], std=[0.1667, 0.0562, 0.1423])
    im_transform = t.Compose([t.ToTensor(), normalize])

    # Use  the following code fo co_transformations e.g. random rotation or random flip etc.
    # co_transformer = cot.Compose([cot.RandomRotate(45)])

    dsetTrain = GIANA(args.imgdir, args.gtdir, input_size=(args.input_width, args.input_height) ,train=True, transform=im_transform, co_transform=None, target_transform=t.ToLabel())
    train_data_loader = data.DataLoader(dsetTrain, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)

    dsetVal = GIANA(args.imgdir, args.gtdir, train=False, transform=im_transform, co_transform=None, target_transform=t.ToLabel())
    val_data_loader = data.DataLoader(dsetVal, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,drop_last=True)
    return train_data_loader, val_data_loader


def train(args, model):
    board = Dashboad(args.visdom_port)
    tr_losses = AverageMeter()
    tr_accs = AverageMeter()
    tLoader, vLoader = load_data(args)
    tr_f1_losses = AverageMeter()
    tr_iou_losses = AverageMeter()
    tr_edge_losses = AverageMeter()
    epoch_iou_loss_tr_list = []
    epoch_f1_loss_tr_list = []
    epoch_edge_loss_tr_list = []
    epoch_loss_tr_list = []
    epoch_acc_tr_list = []
    epoch_acc_vl_list = []

    # #  查看参数量方法一
    # DNN_printer(model, (3, 512, 512), args.batch_size)
    # #  查看参数量方法二
    # summary(model, input_size=(3, 512, 512))   # summary(model, input_size=(ch, h, w), batch_size=-1)

    # input = torch.randn(1, 3, 512, 512)
    # Flops, params = profile(model, inputs=(input.cuda(),))
    # print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
    # print('params参数量: % .4fM'% (params / 1000000))
    # 查看参数量方法三及模型大小
    _, param_sum, _, _, all_size =getModelSize(model)  # 和summary输出的大小略有不同
    print('参数量为：{:.3f}M'.format(param_sum/1e6))

    criterion = nn.BCELoss()  # 使用BCE之前，需要将输出变量量化在[0，1]之间（可以使用Sigmoid激活函数）
    dual_task = DualTaskLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.99)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 表示每隔step_size个epoch就将学习率降为原来的gamma倍
    for epoch in range(1, args.num_epochs+1):
        since = time.time()
        scheduler.step()
        if epoch == 1:
            tr_loss, _, _, _, _, tr_acc = evaluate(args, model, tLoader)
            vl_loss, _, _, vl_fscore, vl_miou, vl_acc = evaluate(args, model, vLoader)

            # Draw the loss curves
            win = None
            win1 = None
            print('[Initial TrainLoss: {0:.4f}]'
                  '\t[Initial TrainAcc: {1:.4f}]'
                  '\t[Initial ValidationLoss: {2:.4f}]'
                  '\t[Initial ValidationFscore: {3:.4f}]'
                  '\t[Initial ValidationMiou: {4:.4f}]'
                  '\t[Initial ValidationAcc: {5:.4f}]'.format(tr_loss, tr_acc, vl_loss, vl_fscore, vl_miou,vl_acc))
            print('----------------------------------------------------------------------------------------------------'
                  '--------------')
        iou_loss_tr_list = []
        f1_loss_tr_list = []
        edge_loss_tr_list = []
        loss_tr_list = []
        acc_tr_list = []
        acc_vl_list = []
        for step, (images, labels) in enumerate(tLoader):
            model.train(True)  # 设置模型状态为训练状态
            loss = 0
            loss_f1 = 0
            loss_edge = 0
            loss_iou = 0
            acc = 0

            if args.cuda:
                images = images.cuda()  # (512, 512)
                labels = labels.cuda()  # (512, 512)

            inputs = Variable(images)  # 将tensor数据类型封装成variable数据类型作为模型的输入
            targets = Variable(labels)  # torch.Size([1, 512, 512]),值：0,1
            targets.data = torch.unsqueeze(targets.data, 1) # torch.Size([1, 512, 512])————>torch.Size([1, 1, 512, 512])
            optimizer.zero_grad()  # 将网络中的所有梯度置0

            if args.add_module == 'none':
                # 网络的前向传播
                outputs = model(inputs)  # N,C,H,W:torch.Size([1, 1, 512, 512]) ,值为[0,1]
                loss = calc_loss(outputs, targets)
                # 训练中得到loss后要回传损失。测试时候只有forward过程
                _, _, acc = Evaluation().specificity_sensitivity(outputs.cpu().data.numpy().squeeze(),
                                                                 targets.cpu().data.numpy())

            elif args.add_module == 'SA':
                # 使用形状注意力，网络两个输出，一个segmentation,一个输出由形状流预测的类边界，[0,1]
                seg_out, edge_out = model(inputs)  # torch.Size([1, 1, 512, 512]) 0-1
                B, C, H, W = seg_out.shape
                # Canny Edge
                targets1 = (labels * 255).cpu().detach().numpy()  # (1, 1, 512, 512),0-255,float32
                targets1 = targets1.astype((np.uint8)).reshape(B * C * H, W)
                targets_edge = cv2.Canny(targets1, 100, 200, 5)
                targets_edge1 = torch.from_numpy(targets_edge / 255).float().cuda()
                targets_edge1 = targets_edge1.reshape(B, C, H, W)

                weight = torch.tensor([1.0, 4.3]).float().cuda()
                # 计算加权系数，并传递给 BCELoss 的 weight 参数
                weight_tensor = torch.ones_like(targets_edge1)
                weight_tensor[targets_edge1 == 1] = weight[1]
                weight_tensor[targets_edge1 == 0] = weight[0]
                loss_edge = F.binary_cross_entropy_with_logits(edge_out, targets_edge1, weight=weight_tensor)

                loss_f1 = Focal_Loss(seg_out, targets)
                loss_iou = iouloss(seg_out, targets)
                loss = loss_f1 + loss_iou + 20 * loss_edge

                # 训练中得到loss后要回传损失。测试时候只有forward过程
                _, _, acc = Evaluation().specificity_sensitivity(seg_out.cpu().data.numpy().squeeze(),
                                                                 targets.cpu().data.numpy())


                # out_edge = out_edge.detach().cpu().numpy()
                # out_edge= out_edge.squeeze()  # (512,512) float32
                # cv2.imwrite(os.path.join('canny/binary_edge/%d.tif' % step), out_edge)

            elif args.add_module == 'SDM':
                # 使用SDM损失，网络两个输出，一个segmentation,一个输出预测的sdm，[-1,1]
                out_tanh, out_seg = model(inputs)  # torch.Size([1, 1, 512, 512])  out_tanh:tensor -1-1,float32
                gt_dis = compute_sdf(targets[:].cpu().numpy(), targets[:, 0, ...].shape)  # numpy: (1, 512, 512),(B,H,W),float64,-1-1
                gt_dis1 = torch.from_numpy(gt_dis).float().cuda()
                mse_loss = MSELoss()
                loss_sdf = mse_loss(out_tanh[:, 0, ...], gt_dis1)
                loss_seg = calc_loss(out_seg, targets)
                loss = loss_seg + 0.1 * loss_sdf
                # 训练中得到loss后要回传损失。测试时候只有forward过程
                _, _, acc = Evaluation().specificity_sensitivity(out_seg.cpu().data.numpy().squeeze(),
                                                                 targets.cpu().data.numpy())

                # 输出SDM
                # gt_dis = gt_dis.squeeze().astype(np.float32)  # (512,512) float32
                # out_tanh = out_tanh.detach().cpu().numpy()
                # out_tanh= out_tanh.squeeze()  # (512,512) float32
                # cv2.imwrite(os.path.join('sdm/output/%d.tif' % step), gt_dis)
                # cv2.imwrite(os.path.join('sdm/target/%d.tif' % step), out_tanh)

            elif args.add_module == 'SDM and SA':
                # 使用形状注意力和SDM
                seg_out, out_tanh, edge_out = model(inputs)  # torch.Size([1, 1, 512, 512]) 0-1
                # sdf
                B, C, H, W = seg_out.shape
                gt_dis = compute_sdf(targets[:].cpu().numpy(), targets[:, 0, ...].shape)  # numpy: (1, 512, 512),(B,H,W),float64,-1-1
                gt_dis1 = torch.from_numpy(gt_dis).float().cuda()
                # Canny Edge
                targets1 = (labels * 255).cpu().detach().numpy()  # (1, 1, 512, 512),0-255,float32
                targets1 = targets1.astype((np.uint8)).reshape(B * C * H, W)
                targets_edge = cv2.Canny(targets1, 100, 200, 5)
                targets_edge1 = torch.from_numpy(targets_edge / 255).float().cuda()
                targets_edge1 = targets_edge1.reshape(B, C, H, W)
                mse_loss = MSELoss()
                loss_sdf = mse_loss(out_tanh[:, 0, ...], gt_dis1)
                loss_edge = criterion(edge_out, targets_edge1)
                loss_seg = calc_loss(seg_out, targets)
                loss = loss_seg + loss_edge + 0.2 * loss_sdf
                # 训练中得到loss后要回传损失。测试时候只有forward过程
                _, _, acc = Evaluation().specificity_sensitivity(seg_out.cpu().data.numpy().squeeze(),
                                                                 targets.cpu().data.numpy())

            loss.backward() # 反向传播求梯度
            optimizer.step()  # 回传损失过程中会计算梯度，然后需要根据这些梯度更新参数，optimizer.step()用来更新参数
            # print("##################", loss.data.cpu().numpy().shape)
            tr_losses.update(loss.data.cpu().numpy())
            tr_f1_losses.update(loss_f1.data.cpu().numpy())
            tr_iou_losses.update(loss_iou.data.cpu().numpy())
            tr_edge_losses.update(loss_edge.data.cpu().numpy())
            tr_accs.update(acc)

            f1_loss_tr_list.append(tr_f1_losses.avg)
            iou_loss_tr_list.append(tr_iou_losses.avg)
            edge_loss_tr_list.append(tr_edge_losses.avg)
            loss_tr_list.append(tr_losses.avg)
            acc_tr_list.append(tr_accs.avg)
        epoch_f1_loss_tr_list.append(sum(f1_loss_tr_list)/len(f1_loss_tr_list))
        epoch_iou_loss_tr_list.append(sum(iou_loss_tr_list)/len(iou_loss_tr_list))
        epoch_edge_loss_tr_list.append(sum(edge_loss_tr_list)/len(edge_loss_tr_list))
        epoch_loss_tr_list.append(sum(loss_tr_list)/len(loss_tr_list))
        epoch_acc_tr_list.append(sum(acc_tr_list)/len(acc_tr_list))

        if epoch % args.log_step == 0:
            vl_loss, _, _, vl_fscore, vl_miou, vl_acc = evaluate(args, model, vLoader)
            acc_vl_list.append(vl_acc)

            print('[Epoch: {0:02}/{1:02}]'
                  '\t[TrainLoss: {2:.4f}]'
                  '\t[TrainAcc: {3:.4f}]'
                  '\t[ValidationLoss: {4:.4f}]'
                  '\t[ValidationFscore: {5:.4f}]'
                  '\t[ValidationMiou: {6:.4f}]'
                  '\t[ValidationAcc: {7:.4f}]'.format(epoch, args.num_epochs, tr_losses.avg, tr_accs.avg, vl_loss, vl_fscore,
                                                       vl_miou, vl_acc)),

            filename = "weights/test/{0}-{1:02}.pth".format(args.model, epoch)
            torch.save(model.state_dict(), filename)
            print('  [Snapshot]')
        else:
            vl_loss, _, _, vl_fscore, vl_miou, vl_acc = evaluate(args, model, vLoader)
            print('[Epoch: {0:02}/{1:02}]'
                  '\t[TrainLoss: {2:.4f}]'
                  '\t[TrainAcc: {3:.4f}]'
                  '\t[ValidationLoss: {4:.4f}]'
                  '\t[ValidationFscore: {5:.4f}]'
                  '\t[ValidationMiou: {6:.4f}]'
                  '\t[ValidationAcc: {7:.4f}]'.format(epoch, args.num_epochs, tr_losses.avg, tr_accs.avg, vl_loss,
                                                      vl_fscore,
                                                      vl_miou, vl_acc)),

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # --- Update the loss curves ---
        win = board.loss_curves([tr_losses.avg, vl_loss], epoch, win=win)
        win1 = board.acc_curves([tr_accs.avg, vl_acc], epoch, win=win1)

    gc.collect()
    torch.cuda.empty_cache()
    json_data = {
        "epoch_f1_loss_tr_list": epoch_f1_loss_tr_list,
        "epoch_iou_loss_tr_list": epoch_iou_loss_tr_list,
        "epoch_edge_loss_tr_list": epoch_edge_loss_tr_list,
        "epoch_loss_tr_list": epoch_loss_tr_list,
        "epoch_acc_tr_list": epoch_acc_tr_list,
        "epoch_acc_vl_list": epoch_acc_vl_list,
    }
    with open("save_json/test/训练过程数据.json", "w") as json_file:
        json.dump(json_data, json_file)


def evaluate(args, model, val_loader):
    model.eval()
    losses = AverageMeter()
    jaccars = AverageMeter()
    dices = AverageMeter()
    accs = AverageMeter()
    mIoUs = AverageMeter()
    f_scores = AverageMeter()
    eva = Evaluation()
    dual_task = DualTaskLoss()

    weight = torch.tensor([1.0, 4.3]).float()
    criterion = nn.BCELoss()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                # criterion = criterion.cuda()

            images = Variable(images)  # torch.Size([1, 3, 512, 512])
            labels = Variable(labels)  # torch.Size([1, 512, 512])
            labels.data = torch.unsqueeze(labels.data, 1)  # torch.Size([1, 512, 512])——->torch.Size([1, 1, 512, 512])

            if args.add_module == 'none':
                outputs = model(images)  # torch.Size([1, 1, 512, 512])
                loss = calc_loss(outputs, labels)
                losses.update(loss.data.cpu().numpy())

                jacc_index = eva.jaccard_similarity_coefficient(outputs.cpu().data.numpy().squeeze(),
                                                                labels.cpu().data.numpy())
                dice_index = eva.dice_coefficient(outputs.cpu().data.numpy().squeeze(),
                                                  labels.cpu().data.numpy())
                f_score_index, mIoU_index, acc_index = eva.specificity_sensitivity(outputs.cpu().data.numpy().squeeze(),
                                                                                   labels.cpu().data.numpy())
            elif args.add_module == 'SA':
                # 使用形状注意力，网络两个输出，一个segmentation,一个输出由形状流预测的类边界，[0,1]
                seg_out, edge_out = model(images)  # torch.Size([1, 1, 512, 512]) 0-1
                B, C, H, W = seg_out.shape
                seg_size = seg_out.size()
                # Canny Edge
                labels1 = (labels*255).cpu().detach().numpy()  # (1, 1, 512, 512),0-255,float32
                labels1 = labels1.astype((np.uint8)).reshape(B*C*H, W)
                labels_edge = cv2.Canny(labels1, 100, 200, 5)
                labels_edge = torch.from_numpy(labels_edge/255).float().cuda()
                labels_edge = labels_edge.reshape(B, C, H, W)

                # 计算加权系数，并传递给 BCELoss 的 weight 参数
                weight_tensor = torch.ones_like(labels_edge)
                weight_tensor[labels_edge == 1] = weight[1]
                weight_tensor[labels_edge == 0] = weight[0]
                loss_edge = F.binary_cross_entropy_with_logits(edge_out, labels_edge, weight=weight_tensor)

                loss_f1 = Focal_Loss(seg_out, labels)
                loss_iou = iouloss(seg_out, labels)
                loss = loss_f1 + loss_iou + 20 * loss_edge


                losses.update(loss.data.cpu().numpy())
                jacc_index = eva.jaccard_similarity_coefficient(seg_out.cpu().data.numpy().squeeze(),
                                                                labels.cpu().data.numpy())
                dice_index = eva.dice_coefficient(seg_out.cpu().data.numpy().squeeze(),
                                                  labels.cpu().data.numpy())
                f_score_index, mIoU_index, acc_index = eva.specificity_sensitivity(seg_out.cpu().data.numpy().squeeze(),labels.cpu().data.numpy())

            elif args.add_module == 'SDM':
                # 使用SDM损失，网络两个输出，一个segmentation,一个输出预测的sdm，[-1,1]
                out_tanh, out_seg = model(images)  # torch.Size([1, 1, 512, 512]) 0-1
                # calculate the loss
                with torch.no_grad():
                    gt_dis = compute_sdf(labels[:].cpu().numpy(), labels[:, 0, ...].shape)
                    gt_dis = torch.from_numpy(gt_dis).float().cuda()  # torch.float32
                mse_loss = MSELoss()
                loss_sdf = mse_loss(out_tanh[:, 0, ...], gt_dis)
                # loss_seg = criterion(outputs, labels)
                # loss_seg = iouloss(outputs, labels)
                # loss_seg = Focal_Loss(outputs, labels)
                loss_seg = calc_loss(out_seg, labels)
                loss = 0.5 * loss_seg + 0.5 * loss_sdf

                losses.update(loss.data.cpu().numpy())
                jacc_index = eva.jaccard_similarity_coefficient(out_seg.cpu().data.numpy().squeeze(),
                                                                labels.cpu().data.numpy())
                dice_index = eva.dice_coefficient(out_seg.cpu().data.numpy().squeeze(),
                                                                labels.cpu().data.numpy())
                f_score_index, mIoU_index, acc_index = eva.specificity_sensitivity(out_seg.cpu().data.numpy().squeeze(),
                                                                labels.cpu().data.numpy())

            elif args.add_module == 'SDM and SA':
                # 使用形状注意力和sdf
                seg_out, out_tanh, edge_out = model(images)  # torch.Size([1, 1, 512, 512]) 0-1
                # calculate the loss
                with torch.no_grad():
                    # sdf
                    gt_dis = compute_sdf(labels[:].cpu().numpy(), labels[:, 0, ...].shape)
                    gt_dis = torch.from_numpy(gt_dis).float().cuda()  # torch.float32
                    # Canny Edge
                    B, C, H, W = seg_out.shape
                    labels1 = (labels * 255).cpu().detach().numpy()  # (1, 1, 512, 512),0-255,float32
                    labels1 = labels1.astype((np.uint8)).reshape(B * C * H, W)
                    labels_edge = cv2.Canny(labels1, 100, 200, 5)
                    labels_edge = torch.from_numpy(labels_edge / 255).float().cuda()
                    labels_edge = labels_edge.reshape(B, C, H, W)
                mse_loss = MSELoss()
                loss_sdf = mse_loss(out_tanh[:, 0, ...], gt_dis)
                loss_edge = criterion(edge_out, labels_edge)
                loss_seg = calc_loss(seg_out, labels)
                loss = loss_seg + loss_edge + 0.2 * loss_sdf
                losses.update(loss.data.cpu().numpy())
                jacc_index = eva.jaccard_similarity_coefficient(seg_out.cpu().data.numpy().squeeze(),
                                                                labels.cpu().data.numpy())
                dice_index = eva.dice_coefficient(seg_out.cpu().data.numpy().squeeze(),
                                                  labels.cpu().data.numpy())
                f_score_index, mIoU_index, acc_index = eva.specificity_sensitivity(seg_out.cpu().data.numpy().squeeze(),
                                                                                   labels.cpu().data.numpy())



            jaccars.update(jacc_index)
            dices.update(dice_index)
            f_scores.update(f_score_index)
            mIoUs.update(mIoU_index)
            accs.update(acc_index)
    return losses.avg, jaccars.avg, dices.avg, f_scores.avg, mIoUs.avg, accs.avg
#
# def eval(args, model):
#
#     if not os.path.exists(args.savedir):
#         os.makedirs(args.savedir)
#
#     listImgFiles = []
#     for k in glob.glob(os.path.join(args.imgdir, '*.tif')):
#         a = os.path.basename(k)  # 带后缀的文件名
#         b = a.split('.')[0]
#         listImgFiles.append(b)
#     for currFile in tqdm(listImgFiles):
#         img = cv2.imread(os.path.join(args.imgdir, currFile + '.tif'), cv2.IMREAD_UNCHANGED)
#         img = np.uint8(img * 255)
#         cv2.imwrite(os.path.join(args.imgdir, currFile + '.tif'), img)
#         img = Image.fromarray(img)
#         img = t.ToTensor()(img)
#         if args.cuda:
#             img.cuda()
#         # saunet
#         output, edge_out, g1, g2, g3, g4, g5 = model(Variable(img).unsqueeze(0))
#         output = t.ToPILImage()(output[0].cpu().data)  # 转为0-255的PIL image
#         output = np.array(output)
#         import matplotlib.pyplot as plt
#
#         g1 = g1.squeeze()
#
#
#         g1 = t.ToPILImage()(g1[0].cpu().data)  # 转为0-255的PIL image
#         g1 = np.array(g1)  # (32,32,3)
#         g2 = t.ToPILImage()(g2[0].cpu().data)  # 转为0-255的PIL image
#         g2 = np.array(g2)
#         g3 = t.ToPILImage()(g3[0].cpu().data)  # 转为0-255的PIL image
#         g3 = np.array(g3)
#         g4 = t.ToPILImage()(g4[0].cpu().data)  # 转为0-255的PIL image
#         g4 = np.array(g4)
#         g5 = t.ToPILImage()(g5[0].cpu().data)  # 转为0-255的PIL image
#         g5 = np.array(g5)
#
#         output[output > 127] = 255
#         output[output <= 127] = 0
#         newfilename = os.path.join(args.savedir, currFile + '.tif')
#         newfilename1 = os.path.join(r"D:\6_code\binary_seg\binaryseg_pytoch\binseg_pytoch-master\data\hh_dot_hv\results\add_experiments\mid\aspp_out\conv1", currFile + '.tif')
#         newfilename2 = os.path.join(r"D:\6_code\binary_seg\binaryseg_pytoch\binseg_pytoch-master\data\hh_dot_hv\results\add_experiments\mid\aspp_out\conv2", currFile + '.tif')
#         newfilename3 = os.path.join(r"D:\6_code\binary_seg\binaryseg_pytoch\binseg_pytoch-master\data\hh_dot_hv\results\add_experiments\mid\aspp_out\conv3", currFile + '.tif')
#         newfilename4 = os.path.join(r"D:\6_code\binary_seg\binaryseg_pytoch\binseg_pytoch-master\data\hh_dot_hv\results\add_experiments\mid\aspp_out\conv4",currFile + '.tif')
#         newfilename5 = os.path.join(r"D:\6_code\binary_seg\binaryseg_pytoch\binseg_pytoch-master\data\hh_dot_hv\results\add_experiments\mid\aspp_out\global",currFile + '.tif')
#
#         cv2.imwrite(newfilename, output)
#         cv2.imwrite(newfilename1, g1[:,:,0])
#         cv2.imwrite(newfilename2, g2[:,:,0])
#         cv2.imwrite(newfilename3, g3[:,:,0])
#         cv2.imwrite(newfilename4, g4[:,:,0])
#         cv2.imwrite(newfilename5, g5[:,:,0])
#
#
#
#     print('success!')
#     time_elapsed = time.time() - start
#     print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def eval(args, model):

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    listImgFiles = []
    for k in glob.glob(os.path.join(args.imgdir, '*.tif')):
        a = os.path.basename(k)  # 带后缀的文件名
        b = a.split('.')[0]
        listImgFiles.append(b)
    for currFile in tqdm(listImgFiles):
        img = cv2.imread(os.path.join(args.imgdir, currFile + '.tif'), cv2.IMREAD_UNCHANGED)
        img = np.uint8(img * 255)
        cv2.imwrite(os.path.join(args.imgdir, currFile + '.tif'), img)
        img = Image.fromarray(img)
        img = t.ToTensor()(img)
        if args.cuda:
            img.cuda()
        # # saunet
        output, _ = model(Variable(img).unsqueeze(0))
        # 单输出
        # output = model(Variable(img).unsqueeze(0))
        # # sdf
        # _, output = model(Variable(img).unsqueeze(0))
        output = t.ToPILImage()(output[0].cpu().data)  # 转为0-255的PIL image
        output = np.array(output)
        output[output > 127] = 255
        output[output <= 127] = 0
        newfilename = os.path.join(args.savedir, currFile + '.tif')
        cv2.imwrite(newfilename, output)


    print('success!')
    time_elapsed = time.time() - start
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def make_model(args):
    Net = None
    if args.model == 'unet':
        Net = UNet
    elif args.model == 'unet_2':
        Net = UNet_2
    elif args.model == 'unet_ensa':
        Net = UNet_EnSA
    elif args.model == 'transunet':
        Net = TransUNet
    elif args.model == 'transunet_2':
        Net = TransUNet_2
    elif args.model == 'deeplab_v3_plus_2':
        Net = DeepLabv3_plus_2
    elif args.model == 'deeplab_v3_plus_sa':
        Net = DeepLabv3_plus_SA
    elif args.model == 'deeplab_v3_plus_2_res50':
        Net = DeepLabv3_plus_2_res50
    elif args.model == 'deeplab_v3_plus':
        Net = DeepLabv3_plus
    elif args.model == 'deeplab_v3_plus_res50':
        Net = DeepLabv3_plus_res50
    elif args.model == 'deeplab_v3_plus_sa':
        Net = DeepLabv3_plus_SA
    elif args.model == 'deeplab_v3_plus_secpam_sa1_res101':
        Net = DeepLabv3_plus_SECPAM_SA1_RES101
    elif args.model == 'deeplab_v3_plus_se':
        Net = DeepLabv3_plus_SE
    elif args.model == 'deeplab_v3_plus_cbam_sa1_res101':
        Net = DeepLabv3_plus_CBAM_SA1
    elif args.model == 'deeplab_v3_plus_secpam_sa1_res101':
        Net = DeepLabv3_plus_SECPAM_SA1_RES101
    elif args.model == 'deeplab_v3_plus_pam_lau_res50':
        Net = DeepLabv3_plus_PAM_LAU_res50
    elif args.model == 'deeplab_v3_plus_lau_se_res50':
        Net = DeepLabv3_plus_LAU_SE_res50
    elif args.model == 'deeplab_v3_plus_lau_asppr_res50':
        Net = DeepLabv3_plus_LAU_res50_ASPPr
    elif args.model == 'deeplab_v3_plus_se_pam_sa1_lau_res50':
        Net = DeepLabv3_plus_SE_PAM_SA1_LAU_res50
    elif args.model == 'deeplab_v3_plus_se_lau_res50':
        Net = DeepLabv3_plus_SE_LAU_res50
    elif args.model == 'deeplab_v3_plus_se_lau_allin_res50':
        Net = DeepLabv3_plus_SE_LAU_allin_res50
    elif args.model == 'deeplab_v3_plus_pam_sa1_lau_res50':
        Net = DeepLabv3_plus_PAM_SA1_LAU_res50
    elif args.model == 'deeplab_v3_plus_sa1':
        Net = DeepLabv3_plus_SA1
    elif args.model == 'deeplab_v3_plus_lau_res50':
        Net = DeepLabv3_plus_LAU_res50
    elif args.model == 'deeplab_v3_plus_sa1_res50':
        Net = DeepLabv3_plus_SA1_res50
    elif args.model == 'deeplab_v3_plus_se_sa1_res50':
        Net = DeepLabv3_plus_SE_SA1
    elif args.model == 'deeplab_v3_plus_pam_sa1_res50':
        Net = DeepLabv3_plus_PAM_SA1_res50
    elif args.model == 'deeplab_v3_plus_se_sa1_res50':
        Net = DeepLabv3_plus_SE_SA1_res50
    elif args.model == 'deeplab_v3_plus_se_sa1_pam_res50':
        Net = DeepLabv3_plus_SE_SA1_PAM_res50
    elif args.model == 'deeplab_v3_plus_se_sa_pam_res50':
        Net = DeepLabv3_plus_SE_SA_PAM_res50
    elif args.model == 'deeplab_v3_plus_mix_sa1_res50':
        Net = DeepLabv3_plus_Mix_SA1_res50
    elif args.model == 'deeplab_v3_plus_se_sa1_pam_res50_hh':
        Net = DeepLabv3_plus_SE_SA1_PAM_res50_hh
    elif args.model == 'deeplab_v3_plus_se_sa1_pam_res50_hv':
        Net = DeepLabv3_plus_SE_SA1_PAM_res50_hv
    elif args.model == 'deeplab_v3_plus_se_sa1_pam_res50_4':
        Net =  DeepLabv3_plus_SE_SA1_PAM_res50_4
    elif args.model == 'deeplab_v3_plus_se_sa1_pam_res50_2':
        Net = DeepLabv3_plus_SE_SA1_PAM_res50_2
    elif args.model == 'deeplab_v3_plus_se_sa1_2':
        Net = DeepLabv3_plus_SE_SA1_2
    elif args.model == 'deeplab_v3_plus_mobilev2':
        Net = DeepLabv3_plus_mobilev2
    elif args.model == 'deeplab_v3_plus_mobilev2_se':
        Net = DeepLabv3_plus_mobilev2_SE
    elif args.model == 'deeplab_v3_plus_mobilev2_sa1':
        Net = DeepLabv3_plus_mobilev2_SA1
    elif args.model == 'deeplab_v3_plus_mobilev2_lau':
        Net = DeepLabv3_plus_mobilev2_LAU
    elif args.model == 'deeplab_v3_plus_mobilev2_gaspp':
        Net = DeepLabv3_plus_mobilev2_GASPP
    elif args.model == 'deeplab_v3_plus_mobilev2_gaspp_new':
        Net = DeepLabv3_plus_mobilev2_GASPP_new
    elif args.model == 'deeplab_v3_plus_mobilev2_segaspp_new':
        Net = DeepLabv3_plus_mobilev2_SEGASPP_new
    elif args.model == 'deeplab_v3_plus_mobilev2_segaspp_lau_new':
        Net = DeepLabv3_plus_mobilev2_SEGASPP_LAU_new
    elif args.model == 'deeplab_v3_plus_mobilev2_segaspp_sa1_new':
        Net = DeepLabv3_plus_mobilev2_SEGASPP_SA1_new
    elif args.model == 'deeplab_v3_plus_mobilev2_segaspp_sa1_lau':
        Net = DeepLabv3_plus_mobilev2_SEGASPP_SA1_LAU
    elif args.model == 'deeplab_v3_plus_mobilev2_segaspp_sa1_lau_new':
        Net = DeepLabv3_plus_mobilev2_SEGASPP_SA1_LAU_new
    elif args.model == 'deeplab_v3_plus_mobilev2_aspp':
        Net = DeepLabv3_plus_mobilev2_aspp
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa_lau_asppr':
        Net = DeepLabv3_plus_mobilev2_SE_SA_LAU_ASPPr
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_LAU
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau_aspplite':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_LAU_ASPPLite
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa_lau':
        Net = DeepLabv3_plus_mobilev2_SE_SA_LAU
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau_aspp1':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_LAU_ASPP1
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_aspp1':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_ASPP1
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1':
        Net = DeepLabv3_plus_mobilev2_SE_SA1
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau_aspp1_depth':
        Net = DeepLabv3_plus_mobilev2_SA1_LAU_ASPP1_depth
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau_aspp1_dsc':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_LAU_ASPP1_Dsc
    elif args.model == 'deeplab_v3_plus_mobilev2plus_se_sa1_lau_aspp1':
        Net = DeepLabv3_plus_mobilev2plus_SE_SA1_LAU_ASPP1
    elif args.model == 'deeplab_v3_plus_mobilev2plus_se_sa1_lau_aspp1':
        Net = DeepLabv3_plus_mobilev2plus_SE_SA1_LAU_ASPP1
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau_aspp1_sconv':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_LAU_ASPP1_sconv
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau_aspp1_ghost':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_LAU_ASPP1_Ghost
    elif args.model == 'deeplab_v3_plus_mobilev2_se_sa1_lau_aspp1_iresnet':
        Net = DeepLabv3_plus_mobilev2_SE_SA1_LAU_ASPP1_iresnet
    elif args.model == 'deeplab_v3_plus_mobilev2_sa1_lau_aspp1':
        Net = DeepLabv3_plus_mobilev2_SA1_LAU_ASPP1
    elif args.model == 'deeplab_v3_plus_mobilev2_se_saplus_lau_asppplus':
        Net = DeepLabv3_plus_mobilev2_SE_SAplus_LAU_ASPPplus
    elif args.model == 'deeplab_v3_plus_mobilev2_se_saold_lau_asppr':
        Net = DeepLabv3_plus_mobilev2_SE_SA_LAU_ASPPr
    elif args.model == 'deeplab_v3_plus_mobilev2_cbam':
        Net = DeepLabv3_plus_mobilev2_CBAM
    elif args.model == 'deeplab_v3_plus_mobilev2_se_pam':
        Net = DeepLabv3_plus_mobilev2_SE_PAM


    assert Net is not None, 'model {args.model} is not available'

    model = Net(num_classes = 1)

    if args.cuda:
        nGPUs = torch.cuda.device_count()
        model = torch.nn.DataParallel(model, device_ids=range(nGPUs)).cuda()
        model = model.cuda()
        if args.state:
            # # 单卡训练,加module.
            # state_dict = torch.load('./weights/' + args.state)
            # print('load model...')
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for key, value in state_dict.items():
            #     name = 'module.' + key  # add `module.`
            #     new_state_dict[name] = value
            # model.load_state_dict(new_state_dict)
            # 双卡训练
            print('load model...')
            model.load_state_dict(torch.load('./weights/' + args.state))

    # In case to use weights trained on multiple GPUs in CPU mode
    else:
        checkpoint = torch.load('./weights/' + args.state)
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[0:9] + k[16:]  # remove `module.`
            if k[0] == 'f':
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model = model.cpu()
    return model


def main(args):
    torch.backends.cudnn.enabled = False

    model = make_model(args)

    # # 查看模型参数量方法四
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))

    if args.mode == 'train':
        train(args, model)


    elif args.mode == 'eval':
        eval(args, model)


if __name__ == '__main__':
    start = time.time()

    args = get_arguments(sys.argv[1:])
    print(args)

    if args.mode == 'crossval':
        params = list(itertools.product(args.batch_size, args.optimizer, args.momentum, args.lr))
        args.batch_size, args.optimizer, args.momentum, args.lr = params[args.settings_id]

    main(args)

