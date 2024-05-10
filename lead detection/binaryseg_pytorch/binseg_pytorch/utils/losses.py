from torch import nn
from torch.autograd import Variable
from utils.dualtask_utils import *
from utils.dualtask_utils import _one_hot_embedding, _gumbel_softmax_sample

VOID_LABEL = 255
N_CLASSES = 21

def Focal_Loss(inputs, target, alpha=0.75, gamma=2): # alpha=1, gamma=1
    inputs = inputs.flatten()
    target = target.flatten()

    # logpt  = -nn.CrossEntropyLoss()(temp_inputs, temp_target)
    logpt = -F.binary_cross_entropy(inputs, target)
    pt = torch.exp(logpt)
    # pt = 1 / torch.sigmoid(-logpt) - 1
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def iouloss(input, target):
    smooth = 1.
    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1. - ((2. * intersection + smooth) /
                 (iflat.sum() + tflat.sum() + smooth))

def calc_loss(input, target):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    focal = Focal_Loss(input, target)
    dice = iouloss(input, target)
    loss = 0.5 * dice + 0.5 * focal
    return loss


def active_contour_loss(y_true, y_pred, weight=10):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    长度项的计算是通过计算预测的分割掩码 y_pred 的梯度来衡量轮廓的平滑程度，而没有直接使用 y_true。这是因为长度项的目标是优化预测的轮廓，而不是与真实轮廓进行比较。
    长度项的计算是基于预测分割掩码 y_pred 内部的梯度差异，通过计算水平和垂直方向上的梯度，即 delta_r 和 delta_c。通过计算梯度的平方幅值并求取平均值，来衡量预测的轮廓的平滑度。
    区域项中才使用了 y_true，用于计算预测分割掩码与真实分割掩码之间的差异。区域项的目标是尽量使预测的分割掩码接近真实分割掩码，在计算损失时使用了 y_true 的信息。
    因此，在长度项中并没有直接涉及到 y_true，而是通过计算 y_pred 的梯度来评估轮廓的平滑程度。
    '''
    # length term:通过计算 y_pred 在水平和垂直方向上的梯度来衡量预测轮廓的平滑程度。
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient ,size: (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient  size: (B, C, H, W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # 梯度的平方相加得到梯度的平方幅值 size: (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

    # region term:衡量预测分割和真实分割掩码之间的差异
    C_in = torch.ones_like(y_pred)  # C_in 和 C_out 是与 y_pred 形状相同的张量，其中 C_in 填充为全为 1，C_out 填充为全为 0
    C_out = torch.zeros_like(y_pred)

    region_in = torch.mean(y_pred * (y_true - C_in) ** 2)  # 计算轮廓区域内 y_pred 和 C_in 之间的均方差
    region_out = torch.mean((1 - y_pred) * (y_true - C_out) ** 2)  # 计算轮廓区域外 1 - y_pred 和 C_out 之间的均方差
    region = region_in + region_out

    loss = weight * lenth + region

    return loss
def crossentropyloss(logits, label):
    mask = (label.view(-1) != VOID_LABEL)
    nonvoid = mask.long().sum()
    if nonvoid == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    # if nonvoid == mask.numel():
    #     # no void pixel, use builtin
    #     return F.cross_entropy(logits, Variable(label))
    target = label.view(-1)[mask]
    C = logits.size(1)
    logits = logits.permute(0, 2, 3, 1)  # B, H, W, C
    logits = logits.contiguous().view(-1, C)
    mask2d = mask.unsqueeze(1).expand(mask.size(0), C).contiguous().view(-1)
    logits = logits[mask2d].view(-1, C)
    loss = F.cross_entropy(logits, Variable(target))
    return loss


# Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0, batch_weights = False):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = batch_weights

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss


# Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def bce2d(input, target):
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)

    target_trans[pos_index] = 1
    target_trans[neg_index] = 0

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight[ignore_index] = 0

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
    return loss


# 正则化项2
def edge_attention(input, target, edge, train = True):  # seg_out, segmask, edge_out
    filler = torch.ones_like(target) * 255
    if train:
        seg_loss = ImageBasedCrossEntropyLoss2d(classes=1, ignore_index=255, upper_bound=1.0).cuda()
    elif not train:
        seg_loss = CrossEntropyLoss2d(size_average=True, ignore_index=255).cuda()
    return seg_loss(input, torch.where(edge.max(1)[0] > 0.8, target.squeeze(0).long(), filler.squeeze(0).long()))

# 正则化项1
class DualTaskLoss(nn.Module):  # 分割输出seg_out和分割GT segmask
    def __init__(self, cuda=False):
        super(DualTaskLoss, self).__init__()
        self._cuda = cuda
        return

    def forward(self, input_logits, gts, ignore_pixel=255):
        """
        :param input_logits: NxCxHxW
        :param gt_semantic_masks: NxCxHxW
        :return: final loss
        """
        N, C, H, W = input_logits.shape
        th = 1e-8  # 1e-10
        eps = 1e-10
        ignore_mask = (gts == ignore_pixel).detach()
        input_logits = torch.where(ignore_mask.view(N, 1, H, W),
                                   torch.zeros(N, C, H, W).cuda(),
                                   input_logits)
        gt_semantic_masks = gts.detach()
        gt_semantic_masks = torch.where(ignore_mask, torch.zeros(N, H, W).long().cuda(), gt_semantic_masks)  # [1, 1, 512, 512]
        # gt_semantic_masks = _one_hot_embedding(gt_semantic_masks.flatten().long(), 2).detach()



        g = _gumbel_softmax_sample(input_logits.view(N, C, -1), tau=0.5)
        g = g.reshape((N, C, H, W))
        g = compute_grad_mag(g, cuda=self._cuda)  # 对分割输出求梯度幅值

        g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)  # 用GT语义标签求梯度幅值

        g = g.view(N, -1)
        g_hat = g_hat.view(N, -1)
        loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return total_loss

if __name__ == "__main__":
    x = torch.tensor([0., 1., 0.])
    y = torch.tensor([0., 1., 1.])
    loss = Focal_Loss(x, y)
    print(loss)


