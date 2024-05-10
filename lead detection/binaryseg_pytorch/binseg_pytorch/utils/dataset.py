import cv2
import torch
import torchvision.datasets as dst
import torch.utils.data as data
from PIL import Image
import glob
import os
import numpy as np
from random import shuffle
import shutil
from PIL import Image
from tqdm import tqdm

class GIANA(data.Dataset):
    def __init__(self, img_root, gt_root, input_size=(512, 512), train=True, transform=None, target_transform=None, co_transform=None):
        self.img_root = img_root
        self.gt_root = gt_root

        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        self.img_filenames = []
        self.gt_filenames = []
        self.input_width = input_size[0]
        self.input_height = input_size[1]

        # --- Check if the dataset is already partitoned and augmented ---
        # --- Otherwise. first partition the data, and them augment the training set ---
        temp = glob.glob(os.path.join(self.img_root, "train_*.tif"))
        if not temp:
            self.partition_data(0.8)
            self.augment_data()

        if train:  # train
            self.img_filenames = glob.glob(os.path.join(self.img_root, "train_*.tif"))
            self.gt_filenames = glob.glob(os.path.join(self.gt_root, "train_*.tif"))

            self.train_data_im = []
            self.train_data_gt = []
            for fname in self.img_filenames:
                im = cv2.imread(fname,cv2.IMREAD_UNCHANGED)
                self.train_data_im.append(im)
            self.train_data_im = np.stack(self.train_data_im, axis=0)  # 堆叠成三维数组，axis=0表示第一维度

            for fname in self.gt_filenames:
                gt = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
                self.train_data_gt.append(gt)
            self.train_data_gt = np.stack(self.train_data_gt, axis=0)

        else:  # val
            self.img_filenames = glob.glob(os.path.join(self.img_root, "val_*.tif"))
            self.gt_filenames = glob.glob(os.path.join(self.gt_root, "val_*.tif"))
            self.val_data_im = []
            self.val_data_gt = []
            for fname in self.img_filenames:
                im = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
                self.val_data_im.append(im)
            self.val_data_im = np.stack(self.val_data_im, axis=0)

            for fname in self.gt_filenames:
                gt = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
                self.val_data_gt.append(gt)
            self.val_data_gt = np.stack(self.val_data_gt, axis=0)


    def __getitem__(self, index):
        im = cv2.imread(self.img_filenames[index],cv2.IMREAD_UNCHANGED)  # H,W,C:(512, 512, 3)
        target = cv2.imread(self.gt_filenames[index],cv2.IMREAD_UNCHANGED)  # (512, 512),有问题
        # im = im.astype(np.uint8)/im = Image.fromarray(np.uint8(im))
        im = Image.fromarray(np.uint8(im * 255))  # numpy.array转PIL.Image
        target = Image.fromarray(np.uint8(target))  # (512, 512)


        # Note: co_transforms must came before ToTensor(), because some functions like flipr does not work on TorchTensor
        # So you should apply the transformations first, and then transform it to TorchTensor
        if self.co_transform is not None:
            im, target = self.co_transform(im, target)
        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        return len(self.img_filenames)

    def partition_data(self, fraction_value):
        filenameList = []
        for k in glob.glob(os.path.join(self.img_root, '*.tif')):
            a = os.path.basename(k)  # 带后缀的文件名
            b = a.split('.')[0]
            filenameList.append(b)
        shuffle(filenameList)

        nSamples = len(filenameList)
        nTraining = int(fraction_value * nSamples)
        nValidation = int((1 - fraction_value) * nSamples)

        dirs = [self.img_root, self.gt_root]
        for index in range(nTraining):
            for d in dirs:
                srcFilename = os.path.join(d, filenameList[index] + '.tif')
                dstFilename = os.path.join(d, 'train_' + filenameList[index] + '.tif')
                shutil.move(srcFilename, dstFilename)

        if nValidation != 0:
            for index in range(nTraining, nTraining + nValidation + 1):
                for d in dirs:
                    srcFilename = os.path.join(d, filenameList[index] + '.tif')
                    dstFilename = os.path.join(d, 'val_' + filenameList[index] + '.tif')
                    shutil.move(srcFilename, dstFilename)

    def augment_data(self, HVFlip=True, HFlip=True, VFlip=True):
        listImgFiles = []
        for k in glob.glob(os.path.join(self.img_root, '*.tif')):
            a = os.path.basename(k)  # 带后缀的文件名
            b = a.split('.')[0]
            listImgFiles.append(b)

        listFilesTrain = [k for k in listImgFiles if 'train' in k]
        listFilesVal = [k for k in listImgFiles if 'train' not in k]

        for filenames in tqdm(listFilesVal):
            src_im_data = cv2.imread(os.path.join(self.img_root, filenames + '.tif'), cv2.IMREAD_UNCHANGED)
            gt_im_data = cv2.imread(os.path.join(self.gt_root, filenames + '.tif'), cv2.IMREAD_UNCHANGED)
            src_im_data = np.uint8(src_im_data * 255)
            gt_im_data =np.uint8(gt_im_data)
            cv2.imwrite(os.path.join(self.img_root, filenames + '.tif'), src_im_data)
            cv2.imwrite(os.path.join(self.gt_root, filenames + '.tif'), gt_im_data)
            # src_im_width, src_im_height, src_im_bands, src_im_data, src_im_geotrans, src_im_proj = readTif(os.path.join(self.img_root, filenames + '.tif'))
            # gt_im_width, gt_im_height, gt_im_bands, gt_im_data, gt_im_geotrans, gt_im_proj = readTif(os.path.join(self.img_root, filenames + '.tif'))
            # src_im_data = np.uint8(src_im_data * 255)
            # gt_im_data =np.uint8(gt_im_data * 255)
            # writeTiff(src_im_data, src_im_geotrans, src_im_proj, os.path.join(self.img_root, filenames + '.tif'))
            # writeTiff(gt_im_data, gt_im_geotrans, gt_im_proj, os.path.join(self.gt_root, filenames + '.tif'))
            if HVFlip:
                hv_im = cv2.flip(src_im_data,-1)
                hv_gt = cv2.flip(gt_im_data, -1)
                cv2.imwrite(os.path.join(self.img_root, filenames + '_hv.tif'), hv_im)
                cv2.imwrite(os.path.join(self.gt_root, filenames + '_hv.tif'), hv_gt)
                # hv_im = np.flip(src_im_data)
                # hv_gt = np.flip(gt_im_data)
                # writeTiff(hv_im, src_im_geotrans, src_im_proj, os.path.join(self.img_root, filenames + '_hv.tif'))
                # writeTiff(hv_gt, gt_im_geotrans, gt_im_proj, os.path.join(self.gt_root, filenames + '_hv.tif'))

            if VFlip:
                vert_im = cv2.flip(src_im_data, 0)
                vert_gt = cv2.flip(gt_im_data, 0)
                cv2.imwrite(os.path.join(self.img_root, filenames + '_vert.tif'), vert_im)
                cv2.imwrite(os.path.join(self.gt_root, filenames + '_vert.tif'), vert_gt)

                # vert_im = np.flip(src_im_data, axis=0)
                # vert_gt = np.flip(gt_im_data, axis=0)
                # writeTiff(vert_im, src_im_geotrans, src_im_proj, os.path.join(self.img_root, filenames + '_vert.tif'))
                # writeTiff(vert_gt, gt_im_geotrans, gt_im_proj, os.path.join(self.gt_root, filenames + '_vert.tif'))
            if HFlip:
                horz_im = cv2.flip(src_im_data, 1)
                horz_gt = cv2.flip(gt_im_data, 1)
                cv2.imwrite(os.path.join(self.img_root, filenames + '_horz.tif'), horz_im)
                cv2.imwrite(os.path.join(self.gt_root, filenames + '_horz.tif'), horz_gt)

        for filenames in tqdm(listFilesTrain):
            src_im_data = cv2.imread(os.path.join(self.img_root, filenames + '.tif'), cv2.IMREAD_UNCHANGED)
            gt_im_data = cv2.imread(os.path.join(self.gt_root, filenames + '.tif'), cv2.IMREAD_UNCHANGED)
            src_im_data = np.uint8(src_im_data * 255)
            gt_im_data = np.uint8(gt_im_data)
            cv2.imwrite(os.path.join(self.img_root, filenames + '.tif'), src_im_data)
            cv2.imwrite(os.path.join(self.gt_root, filenames + '.tif'), gt_im_data)
            # src_im_width, src_im_height, src_im_bands, src_im_data, src_im_geotrans, src_im_proj = readTif(os.path.join(self.img_root, filenames + '.tif'))
            # gt_im_width, gt_im_height, gt_im_bands, gt_im_data, gt_im_geotrans, gt_im_proj = readTif(os.path.join(self.img_root, filenames + '.tif'))
            # src_im_data = np.uint8(src_im_data * 255)
            # gt_im_data = np.uint8(gt_im_data * 255)
            # writeTiff(src_im_data, src_im_geotrans, src_im_proj, os.path.join(self.img_root, filenames + '.tif'))
            # writeTiff(gt_im_data, gt_im_geotrans, gt_im_proj, os.path.join(self.gt_root, filenames + '.tif'))
            if HVFlip:
                hv_im = cv2.flip(src_im_data,-1)
                hv_gt = cv2.flip(gt_im_data, -1)
                cv2.imwrite(os.path.join(self.img_root, filenames + '_hv.tif'), hv_im)
                cv2.imwrite(os.path.join(self.gt_root, filenames + '_hv.tif'), hv_gt)
                # hv_im = np.flip(src_im_data)
                # hv_gt = np.flip(gt_im_data)
                # writeTiff(hv_im, src_im_geotrans, src_im_proj, os.path.join(self.img_root, filenames + '_hv.tif'))
                # writeTiff(hv_gt, gt_im_geotrans, gt_im_proj, os.path.join(self.gt_root, filenames + '_hv.tif'))

            if VFlip:
                vert_im = cv2.flip(src_im_data, 0)
                vert_gt = cv2.flip(gt_im_data, 0)
                cv2.imwrite(os.path.join(self.img_root, filenames + '_vert.tif'), vert_im)
                cv2.imwrite(os.path.join(self.gt_root, filenames + '_vert.tif'), vert_gt)

                # vert_im = np.flip(src_im_data, axis=0)
                # vert_gt = np.flip(gt_im_data, axis=0)
                # writeTiff(vert_im, src_im_geotrans, src_im_proj, os.path.join(self.img_root, filenames + '_vert.tif'))
                # writeTiff(vert_gt, gt_im_geotrans, gt_im_proj, os.path.join(self.gt_root, filenames + '_vert.tif'))
            if HFlip:
                horz_im = cv2.flip(src_im_data, 1)
                horz_gt = cv2.flip(gt_im_data, 1)
                cv2.imwrite(os.path.join(self.img_root, filenames + '_horz.tif'), horz_im)
                cv2.imwrite(os.path.join(self.gt_root, filenames + '_horz.tif'), horz_gt)

                # horz_im = np.flip(src_im_data, axis=1)
                # horz_gt = np.flip(gt_im_data, axis=1)
                # writeTiff(horz_im, src_im_geotrans, src_im_proj, os.path.join(self.img_root, filenames + '_horz.tif'))
                # writeTiff(horz_gt, gt_im_geotrans, gt_im_proj, os.path.join(self.gt_root, filenames + '_horz.tif'))


