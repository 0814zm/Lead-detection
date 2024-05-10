import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils.tools import initialize_weights
from models.resnet import BasicBlock as ResBlock
from models.GSConv import GatedSpatialConv2d


class UnetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                              nn.BatchNorm2d(self.out_channels),
                              nn.ReLU(),
                              nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                              nn.BatchNorm2d(self.out_channels),
                              nn.ReLU())

    def forward(self,x):
        return self.layer(x)


class UnetDecoder(nn.Module):

    def __init__(self, in_channels, featrures, out_channels):
        super(UnetDecoder, self).__init__()
        self.in_channels = in_channels
        self.features = featrures
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.Conv2d(self.features, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(self.features, self.out_channels, kernel_size=2, stride=2),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU())

    def forward(self,x):
        return self.layer(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 #nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))
                                 #nn.BatchNorm2d(64),
                                 # nn.ReLU())

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()


        #Initialize weights
        initialize_weights(self)


    def forward(self, x):
        en1 = self.down1(x)
        po1 = self.pool1(en1)
        en2 = self.down2(po1)
        po2 = self.pool2(en2)
        en3 = self.down3(po2)
        po3 = self.pool3(en3)
        en4 = self.down4(po3)
        po4 = self.pool4(en4)

        c1 = self.center(po4)

        dec1 = self.up1(torch.cat([c1, F.upsample_bilinear(en4, c1.size()[2:])], 1))
        dec2 = self.up2(torch.cat([dec1, F.upsample_bilinear(en3, dec1.size()[2:])], 1))
        dec3 = self.up3(torch.cat([dec2, F.upsample_bilinear(en2, dec2.size()[2:])], 1))
        dec4 = self.up4(torch.cat([dec3, F.upsample_bilinear(en1, dec3.size()[2:])], 1))
        
        out = self.output(dec4)
        return self.final(out)


class UNet_2(nn.Module):
    def __init__(self, num_classes):
        super(UNet_2, self).__init__()
        self.num_classes = num_classes
        self.down1 = UnetEncoder(2, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))
        # nn.BatchNorm2d(64),
        # nn.ReLU())

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        # print('x:', x.shape)
        x = x[:, :2, :, :]  # [1, 1, 512, 512]
        # print('x:', x.shape)
        en1 = self.down1(x)
        po1 = self.pool1(en1)
        en2 = self.down2(po1)
        po2 = self.pool2(en2)
        en3 = self.down3(po2)
        po3 = self.pool3(en3)
        en4 = self.down4(po3)
        po4 = self.pool4(en4)

        c1 = self.center(po4)

        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:])], 1))
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:])], 1))
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:])], 1))
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:])], 1))

        out = self.output(dec4)
        return self.final(out)
class UNet_sdf(nn.Module):
    def __init__(self, num_classes):
        super(UNet_sdf, self).__init__()
        self.num_classes = num_classes
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))
        # nn.BatchNorm2d(64),
        # nn.ReLU())

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.output1 = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        en1 = self.down1(x)
        po1 = self.pool1(en1)
        en2 = self.down2(po1)
        po2 = self.pool2(en2)
        en3 = self.down3(po2)
        po3 = self.pool3(en3)
        en4 = self.down4(po3)
        po4 = self.pool4(en4)

        c1 = self.center(po4)

        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:])], 1))
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:])], 1))
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:])], 1))
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:])], 1))

        out = self.output(dec4)
        out_tanh = self.tanh(out)
        out_seg = self.output1(dec4)
        return out_tanh, self.final(out_seg)


class UNet_EnSA(nn.Module):
    def __init__(self, num_classes):
        super(UNet_EnSA, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)
        self.d3 = nn.Conv2d(8, 4, kernel_size=1)
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(16, 16)
        self.gate2 = GatedSpatialConv2d(8, 8)
        self.gate3 = GatedSpatialConv2d(4, 4)

        # Encoder
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        # Decoder
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        x_size = x.size()

        en1 = self.down1(x)  # [1, 64, 512, 512]
        po1 = self.pool1(en1)
        en2 = self.down2(po1)  # [1, 128, 256, 256]
        po2 = self.pool2(en2)
        en3 = self.down3(po2)  # [1, 256, 128, 128]
        po3 = self.pool3(en3)
        en4 = self.down4(po3)  # [1, 512, 64, 64]
        po4 = self.pool4(en4)  # [1, 512, 32, 32]


        # Shape Stream
        ss = F.interpolate(self.d0(en1), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 32, 512, 512],do的作用64——>32
        ss = self.res1(ss)  # [1, 32, 512, 512]
        c3 = F.interpolate(self.c3(en2), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，128——>1
        ss = self.d1(ss)  # [1, 16, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 16, 512, 512]
        a1 = ss
        ss = self.res2(ss)  # [1, 16, 512, 512]
        ss = self.d2(ss)  # [1, 8, 512, 512]
        c4 = F.interpolate(self.c4(en3), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 8, 512, 512]
        a2 = ss
        ss = self.res3(ss)
        ss = self.d3(ss)  # [1, 4, 512, 512]
        c5 = F.interpolate(self.c5(en4), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 4, 512, 512]
        a3 = ss
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)


        c1 = self.center(po4)  # [1, 512, 64, 64]


        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:])], 1))  # [1, 256, 128, 128]
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:])], 1))  # [1, 128, 256, 256]
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:])], 1))  # [1, 64, 512, 512]
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:])], 1))  # [1, 64, 512, 512]

        out = self.output(dec4)
        return self.final(out), edge_out, g1, g2, g3


class UNet_EnSA1(nn.Module):
    def __init__(self, num_classes):
        super(UNet_EnSA1, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 1, kernel_size=1)
        self.res1 = ResBlock(1, 1)
        self.d1 = nn.Conv2d(1, 1, kernel_size=1)
        self.res2 = ResBlock(1, 1)
        self.d2 = nn.Conv2d(1, 1, kernel_size=1)
        self.res3 = ResBlock(1, 1)
        self.d3 = nn.Conv2d(1, 1, kernel_size=1)
        self.fuse = nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(1, 1)
        self.gate2 = GatedSpatialConv2d(1, 1)
        self.gate3 = GatedSpatialConv2d(1, 1)

        # Encoder
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))

        # Decoder
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        x_size = x.size()

        en1 = self.down1(x)  # [1, 64, 512, 512]
        po1 = self.pool1(en1)
        en2 = self.down2(po1)  # [1, 128, 256, 256]
        po2 = self.pool2(en2)
        en3 = self.down3(po2)  # [1, 256, 128, 128]
        po3 = self.pool3(en3)
        en4 = self.down4(po3)  # [1, 512, 64, 64]
        po4 = self.pool4(en4)

        # Shape Stream
        ss = F.interpolate(self.d0(en1), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 32, 512, 512],do的作用64——>32
        ss = self.res1(ss)  # [1, 32, 512, 512]
        c3 = F.interpolate(self.c3(en2), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，128——>1
        ss = self.d1(ss)  # [1, 16, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 16, 512, 512]
        ss = self.res2(ss)  # [1, 16, 512, 512]
        ss = self.d2(ss)  # [1, 8, 512, 512]
        c4 = F.interpolate(self.c4(en3), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 8, 512, 512]
        ss = self.res3(ss)
        ss = self.d3(ss)  # [1, 4, 512, 512]
        c5 = F.interpolate(self.c5(en4), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 4, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        c1 = self.center(po4)

        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:])], 1))
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:])], 1))
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:])], 1))
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:])], 1))

        out = self.output(dec4)
        return self.final(out), edge_out

# class UNet_DESA(nn.Module):
#     def __init__(self, num_classes, pretrained=True):
#         super(UNet_DESA, self).__init__()
#         self.num_classes = num_classes
#         print("SAUNet w/ Shape Stream")
#         self.pool = nn.MaxPool2d(2, 2)
#         self.encoder = torchvision.models.densenet121(pretrained=pretrained)
#         self.sigmoid = nn.Sigmoid()
#
#         # Shape Stream
#         self.c3 = nn.Conv2d(256, 1, kernel_size=1)
#         self.c4 = nn.Conv2d(512, 1, kernel_size=1)
#         self.c5 = nn.Conv2d(1024, 1, kernel_size=1)
#
#         self.d0 = nn.Conv2d(128, 64, kernel_size=1)
#         self.res1 = ResBlock(64, 64)
#         self.d1 = nn.Conv2d(64, 32, kernel_size=1)
#         self.res2 = ResBlock(32, 32)
#         self.d2 = nn.Conv2d(32, 16, kernel_size=1)
#         self.res3 = ResBlock(16, 16)
#         self.d3 = nn.Conv2d(16, 8, kernel_size=1)
#         self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)
#
#         self.gate1 = GatedSpatialConv2d(32, 32)
#         self.gate2 = GatedSpatialConv2d(16, 16)
#         self.gate3 = GatedSpatialConv2d(8, 8)
#
#         # Encoder
#         self.conv1 = nn.Sequential(self.encoder.features.conv0,
#                                    self.encoder.features.norm0)
#         self.conv2 = self.encoder.features.denseblock1
#         self.conv2t = self.encoder.features.transition1
#         self.conv3 = self.encoder.features.denseblock2
#         self.conv3t = self.encoder.features.transition2
#         self.conv4 = self.encoder.features.denseblock3
#         self.conv4t = self.encoder.features.transition3
#         self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
#                                    self.encoder.features.norm5)
#
#
#
#         self.center = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
#                                     nn.BatchNorm2d(512),
#                                     nn.ReLU())
#
#         # Decoder
#         self.up1 = UnetDecoder(1536, 512, 256)
#         self.up2 = UnetDecoder(768, 256, 128)
#         self.up3 = UnetDecoder(384, 128, 64)
#
#         self.up4 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1),
#                                  # nn.BatchNorm2d(64),
#                                  # nn.ReLU(),
#                                  nn.Conv2d(64, 64, 3, padding=1))
#
#         self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
#         self.final = nn.Sigmoid()
#
#         # Initialize weights
#         initialize_weights(self)
#
#     def forward(self, x):
#         x_size = x.size()
#
#         # Encoder
#         conv1 = self.conv1(x)  # [1, 64, 256, 256]
#         conv2 = self.conv2t(self.conv2(conv1))  # [1, 128, 128, 128]
#         conv3 = self.conv3t(self.conv3(conv2))  # [1, 256, 64, 64]
#         conv4 = self.conv4t(self.conv4(conv3))  # [1, 512, 32, 32]
#         conv5 = self.conv5(conv4)  # [1, 1024, 32, 32]
#
#         # Shape Stream
#         ss = F.interpolate(self.d0(conv2), x_size[2:],
#                            mode='bilinear', align_corners=True)  # [1, 32, 512, 512],do的作用128——>64
#         ss = self.res1(ss)  # [1, 64, 512, 512]
#         c3 = F.interpolate(self.c3(conv3), x_size[2:],
#                            mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
#         ss = self.d1(ss)  # [1, 32, 512, 512]
#         ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 32, 512, 512]
#         ss = self.res2(ss)  # [1, 32, 512, 512]
#         ss = self.d2(ss)  # [1, 16, 512, 512]
#         c4 = F.interpolate(self.c4(conv4), x_size[2:],
#                            mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
#         ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 16, 512, 512]
#         ss = self.res3(ss)  # [1, 16, 512, 512]
#         ss = self.d3(ss)  # [1, 8, 512, 512]
#         c5 = F.interpolate(self.c5(conv5), x_size[2:],
#                            mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
#         ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 8, 512, 512]
#         ss = self.fuse(ss)  # [1, 1, 512, 512]
#         ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
#         edge_out = self.sigmoid(ss)
#
#
#         c1 = self.center(conv5) # [1, 512, 64, 64]
#
#
#         dec1 = self.up1(torch.cat([c1, F.interpolate(conv5, c1.size()[2:])], 1))  # [1, 256, 128, 128]
#         dec2 = self.up2(torch.cat([dec1, F.interpolate(conv4, dec1.size()[2:])], 1))  # [1, 128, 256, 256]
#         dec3 = self.up3(torch.cat([dec2, F.interpolate(conv3, dec2.size()[2:])], 1))  # [1, 64, 512, 512]
#         dec4 = self.up4(torch.cat([dec3, F.interpolate(conv2, dec3.size()[2:])], 1))  # [1, 64, 512, 512]
#
#         out = self.output(dec4)
#         return self.final(out), edge_out

class UNet_DESA(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(UNet_DESA, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)
        self.d3 = nn.Conv2d(8, 4, kernel_size=1)
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(16, 16)
        self.gate2 = GatedSpatialConv2d(8, 8)
        self.gate3 = GatedSpatialConv2d(4, 4)

        # Encoder
        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3



        self.center = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        # Decoder
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        x_size = x.size()

        # Encoder
        conv1 = self.conv1(x)  # [1, 64, 256, 256]
        conv2 = self.conv2t(self.conv2(conv1))  # [1, 128, 128, 128]
        conv3 = self.conv3t(self.conv3(conv2))  # [1, 256, 64, 64]
        conv4 = self.conv4t(self.conv4(conv3))  # [1, 512, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(conv1), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 32, 512, 512],do的作用64——>32
        ss = self.res1(ss)  # [1, 32, 512, 512]
        c3 = F.interpolate(self.c3(conv2), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss = self.d1(ss)  # [1, 16, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 16, 512, 512]
        ss = self.res2(ss)  # [1, 16, 512, 512]
        ss = self.d2(ss)  # [1, 8, 512, 512]
        c4 = F.interpolate(self.c4(conv3), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 8, 512, 512]
        ss = self.res3(ss)  # [1, 8, 512, 512]
        ss = self.d3(ss)  # [1, 4, 512, 512]
        c5 = F.interpolate(self.c5(conv4), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 4, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)


        c1 = self.center(conv4) # [1, 512, 64, 64]


        dec1 = self.up1(torch.cat([c1, F.interpolate(conv4, c1.size()[2:])], 1))  # [1, 256, 128, 128]
        dec2 = self.up2(torch.cat([dec1, F.interpolate(conv3, dec1.size()[2:])], 1))  # [1, 128, 256, 256]
        dec3 = self.up3(torch.cat([dec2, F.interpolate(conv2, dec2.size()[2:])], 1))  # [1, 64, 512, 512]
        dec4 = self.up4(torch.cat([dec3, F.interpolate(conv1, dec3.size()[2:])], 1))  # [1, 64, 512, 512]

        out = self.output(dec4)
        return self.final(out), edge_out

class UNet_DeSA(nn.Module):
    def __init__(self, num_classes):
        super(UNet_DeSA, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(64, 1, kernel_size=1)
        self.c5 = nn.Conv2d(64, 1, kernel_size=1)

        self.d0 = nn.Conv2d(256, 128, kernel_size=1)
        self.res1 = ResBlock(128, 128)
        self.d1 = nn.Conv2d(128, 64, kernel_size=1)
        self.res2 = ResBlock(64, 64)
        self.d2 = nn.Conv2d(64, 32, kernel_size=1)
        self.res3 = ResBlock(32, 32)
        self.d3 = nn.Conv2d(32, 16, kernel_size=1)
        self.fuse = nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(64, 64)
        self.gate2 = GatedSpatialConv2d(32, 32)
        self.gate3 = GatedSpatialConv2d(16, 16)

        # Encoder
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        # Decoder
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        x_size = x.size()

        en1 = self.down1(x)  # [1, 64, 512, 512]
        po1 = self.pool1(en1)
        en2 = self.down2(po1)  # [1, 128, 256, 256]
        po2 = self.pool2(en2)
        en3 = self.down3(po2)  # [1, 256, 128, 128]
        po3 = self.pool3(en3)
        en4 = self.down4(po3)  # [1, 512, 64, 64]
        po4 = self.pool4(en4)

        c1 = self.center(po4)  # [1, 512, 64, 64]


        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:])], 1))  # [1, 256, 128, 128]
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:])], 1))  # [1, 128, 256, 256]
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:])], 1))  # [1, 64, 512, 512]
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:])], 1))  # [1, 64, 512, 512]

        # Shape Stream
        ss = F.interpolate(self.d0(dec1), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(dec2), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss, g1 = self.gate1(ss, c3)  # g1: torch.Size([1, 1, 512, 512])  g1 ss: torch.Size([1, 64, 512, 512])
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(dec3), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss, g2 = self.gate2(ss, c4)  # g2: torch.Size([1, 1, 512, 512]) g2 ss: torch.Size([1, 32, 512, 512])
        ss = self.res3(ss)  # [1, 32, 512, 512]
        ss = self.d3(ss)  # [1, 16, 512, 512]
        c5 = F.interpolate(self.c5(dec4), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3: torch.Size([1, 1, 512, 512]) g3 ss: torch.Size([1, 16, 512, 512])
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # final ss: torch.Size([1, 1, 512, 512])
        edge_out = self.sigmoid(ss)


        out = self.output(dec4)
        return self.final(out), edge_out

class UNet_DeSA1(nn.Module):
    def __init__(self, num_classes):
        super(UNet_DeSA1, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(64, 1, kernel_size=1)
        self.c5 = nn.Conv2d(64, 1, kernel_size=1)

        self.d0 = nn.Conv2d(256, 1, kernel_size=1)
        self.res1 = ResBlock(1, 1)
        self.d1 = nn.Conv2d(1, 1, kernel_size=1)
        self.res2 = ResBlock(1, 1)
        self.d2 = nn.Conv2d(1, 1, kernel_size=1)
        self.res3 = ResBlock(1, 1)
        self.d3 = nn.Conv2d(1, 1, kernel_size=1)
        self.fuse = nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(1, 1)
        self.gate2 = GatedSpatialConv2d(1, 1)
        self.gate3 = GatedSpatialConv2d(1, 1)

        # Encoder
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        # Decoder
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        x_size = x.size()

        en1 = self.down1(x)  # [1, 64, 512, 512]
        po1 = self.pool1(en1)
        en2 = self.down2(po1)  # [1, 128, 256, 256]
        po2 = self.pool2(en2)
        en3 = self.down3(po2)  # [1, 256, 128, 128]
        po3 = self.pool3(en3)
        en4 = self.down4(po3)  # [1, 512, 64, 64]
        po4 = self.pool4(en4)

        c1 = self.center(po4)  # [1, 512, 64, 64]


        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:])], 1))  # [1, 256, 128, 128]
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:])], 1))  # [1, 128, 256, 256]
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:])], 1))  # [1, 64, 512, 512]
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:])], 1))  # [1, 64, 512, 512]

        # Shape Stream
        ss = F.interpolate(self.d0(dec1), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(dec2), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss, g1 = self.gate1(ss, c3)  # g1: torch.Size([1, 1, 512, 512])  g1 ss: torch.Size([1, 1, 512, 512])
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(dec3), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss, g2 = self.gate2(ss, c4)  # g2: torch.Size([1, 1, 512, 512]) g2 ss: torch.Size([1, 1, 512, 512])
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(dec4), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3: torch.Size([1, 1, 512, 512]) g3 ss: torch.Size([1, 1, 512, 512])
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # final ss: torch.Size([1, 1, 512, 512])
        edge_out = self.sigmoid(ss)


        out = self.output(dec4)
        return self.final(out), edge_out

class UNet_DESA1(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(UNet_DESA1, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 1, kernel_size=1)
        self.res1 = ResBlock(1, 1)
        self.d1 = nn.Conv2d(1, 1, kernel_size=1)
        self.res2 = ResBlock(1, 1)
        self.d2 = nn.Conv2d(1, 1, kernel_size=1)
        self.res3 = ResBlock(1, 1)
        self.d3 = nn.Conv2d(1, 1, kernel_size=1)
        self.fuse = nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(1, 1)
        self.gate2 = GatedSpatialConv2d(1, 1)
        self.gate3 = GatedSpatialConv2d(1, 1)

        # Encoder
        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3



        self.center = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))

        # Decoder
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        x_size = x.size()

        # Encoder
        conv1 = self.conv1(x)  # [1, 64, 256, 256]
        conv2 = self.conv2t(self.conv2(conv1))  # [1, 128, 128, 128]
        conv3 = self.conv3t(self.conv3(conv2))  # [1, 256, 64, 64]
        conv4 = self.conv4t(self.conv4(conv3))  # [1, 512, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(conv1), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512],do的作用64——>32
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(conv2), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(conv3), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(conv4), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)


        c1 = self.center(conv4) # [1, 512, 64, 64]


        dec1 = self.up1(torch.cat([c1, F.interpolate(conv4, c1.size()[2:])], 1))  # [1, 256, 128, 128]
        dec2 = self.up2(torch.cat([dec1, F.interpolate(conv3, dec1.size()[2:])], 1))  # [1, 128, 256, 256]
        dec3 = self.up3(torch.cat([dec2, F.interpolate(conv2, dec2.size()[2:])], 1))  # [1, 64, 512, 512]
        dec4 = self.up4(torch.cat([dec3, F.interpolate(conv1, dec3.size()[2:])], 1))  # [1, 64, 512, 512]

        out = self.output(dec4)
        return self.final(out), edge_out
class UNet_EnSA_sdf(nn.Module):
    def __init__(self, num_classes):
        super(UNet_EnSA_sdf, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)
        self.d3 = nn.Conv2d(8, 4, kernel_size=1)
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(16, 16)
        self.gate2 = GatedSpatialConv2d(8, 8)
        self.gate3 = GatedSpatialConv2d(4, 4)

        # Encoder
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        # Decoder
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)  # 1*1卷积将通道数进行调整，调整成所需类别数
        self.output1 = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        x_size = x.size()

        en1 = self.down1(x)  # [1, 64, 512, 512]
        po1 = self.pool1(en1)
        en2 = self.down2(po1)  # [1, 128, 256, 256]
        po2 = self.pool2(en2)
        en3 = self.down3(po2)  # [1, 256, 128, 128]
        po3 = self.pool3(en3)
        en4 = self.down4(po3)  # [1, 512, 64, 64]
        po4 = self.pool4(en4)

        # Shape Stream
        ss = F.interpolate(self.d0(en1), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 32, 512, 512],do的作用64——>32
        ss = self.res1(ss)  # [1, 32, 512, 512]
        c3 = F.interpolate(self.c3(en2), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，128——>1
        ss = self.d1(ss)  # [1, 16, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 16, 512, 512]
        ss = self.res2(ss)  # [1, 16, 512, 512]
        ss = self.d2(ss)  # [1, 8, 512, 512]
        c4 = F.interpolate(self.c4(en3), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 8, 512, 512]
        ss = self.res3(ss)
        ss = self.d3(ss)  # [1, 4, 512, 512]
        c5 = F.interpolate(self.c5(en4), x_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 4, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        c1 = self.center(po4)

        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:])], 1))
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:])], 1))
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:])], 1))
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:])], 1))

        out = self.output(dec4)
        out_tanh = self.tanh(out)
        out_seg = self.output1(dec4)
        seg_out = self.final(out_seg)
        return seg_out, out_tanh, edge_out


if __name__ == "__main__":
    net = UNet_EnSA1(1)
    x1 = torch.rand(1, 3, 512, 512)
    print(net)

    out_seg, out_edge = net(x1)
    print('out_seg.shape:{}'.format(out_seg.shape))  # [1, 1, 512, 512]
    print('out_edge.shaprpe:{}'.format(out_edge.shape))  # [1, 1, 512, 512]










