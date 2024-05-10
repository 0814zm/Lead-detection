import torch
from torch import nn
# 参考https://blog.csdn.net/cp1314971/article/details/104417808


def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer
def conv_block1(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=1)
    )
    return layer
class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition_down(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )
    return trans_layer

def transition_up(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )
    return trans_layer
class EW_Net(nn.Module):
    def __init__(self, num_classes, growth_rate=16, block_layers=[2,4]):
        super(EW_Net, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
            )

        # 熵下采样
        self.conv1 = conv_block(1, 1)
        self.tl1 = self._make_transition_layer_down(1)
        self.conv2 = conv_block(1, 1)
        self.tl2 = self._make_transition_layer_down(1)
        self.conv3 = conv_block(1, 1)
        self.tl3 = self._make_transition_layer_down(1)

        # 图像下采样
        self.DB1 = self._make_dense_block(16, growth_rate,num=block_layers[0])
        self.TL1 = self._make_transition_layer_down(48)
        self.DB2 = self._make_dense_block(48, growth_rate, num=block_layers[0])
        self.TL2 = self._make_transition_layer_down(80)
        self.DB3 = self._make_dense_block(80, growth_rate, num=block_layers[0])
        self.TL3 = self._make_transition_layer_down(112)
        self.DB4 = self._make_dense_block(112, growth_rate, num=block_layers[0])
        self.TL4 = self._make_transition_layer_down(144)

        # 图像上采样
        self.DB5 = self._make_dense_block(144, growth_rate, num=block_layers[1])
        self.TL5 = self._make_transition_layer_up(208, 112)
        self.DB6 = self._make_dense_block(112, growth_rate, num=block_layers[1])
        self.TL6 = self._make_transition_layer_up(176, 80)
        self.DB7 = self._make_dense_block(80, growth_rate, num=block_layers[1])
        self.TL7 = self._make_transition_layer_up(144, 48)
        self.DB8 = self._make_dense_block(48, growth_rate, num=block_layers[1])
        self.TL8 = self._make_transition_layer_up(112, 112)

        self.conv11 = conv_block1(112, num_classes)
        self.final = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.first_block(x1)
        x1d1 = self.DB1(x1)  # [1, 48, 512, 512]
        x1t1 = self.TL1(x1d1)  # [1, 48, 256, 256]
        x2c1 = self.conv1(x2)  # [1, 1, 512, 512]
        x2t1 = self.tl1(x2c1)  # [1, 1, 256, 256]

        x1d2 = self.DB2(x1t1)  # [1, 80, 256, 256]
        x1t2 = self.TL2(x1d2)  # [1, 80, 128, 128]
        x2c2 = self.conv2(x2t1)  # [1, 1, 256, 256]
        x2t2 = self.tl2(x2c2)  # [1, 1, 128, 128]

        x1d3 = self.DB3(x1t2)  # [1, 112, 128, 128]
        x1t3 = self.TL3(x1d3)  # [1, 112, 64, 64]
        x2c3 = self.conv3(x2t2)  # [1, 1, 128, 128]
        x2t3 = self.tl3(x2c3)  # [1, 1, 64, 64]

        x1d4 = self.DB4(x1t3)  # [1, 144, 64, 64]
        x1t4 = self.TL4(x1d4)  # [1, 144, 32, 32]
        x1d5 = self.DB5(x1t4)  # [1, 208, 32, 32]
        x1t5 = self.TL5(x1d5)  # [1, 112, 64, 64]

        x12e1 = self.DB6(torch.add(x1t5, torch.mul(x1t3, self.final(x2t3))))  # [1, 176, 64, 64]
        x1t6 = self.TL6(x12e1)  # [1, 80, 128, 128]

        x12e2 = self.DB7(torch.add(x1t6, torch.mul(x1t2, self.final(x2t2))))  # [1, 144, 128, 128]
        x1t7 = self.TL7(x12e2)  # [1, 48, 256, 256]

        x12e3 = self.DB8(torch.add(x1t7, torch.mul(x1t1, self.final(x2t1))))  # [1, 112, 256, 256]
        x1t8 = self.TL8(x12e3)  # [1, 112, 512, 512]

        out = self.conv11(x1t8)  # [1, 1, 512, 512]
        out = self.final(out)  # [1, 1, 512, 512]

        return out

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer_down(self,channels):
        block = []
        block.append(transition_down(channels, channels))
        return nn.Sequential(*block)

    def _make_transition_layer_up(self, in_channels,out_channels):
        block = []
        block.append(transition_up(in_channels, out_channels))
        return nn.Sequential(*block)


net = EW_Net(1)
x1 = torch.rand(1,3,512,512)
x2 = torch.rand(1,1,512,512)
print(net)

out = net(x1, x2)
print(out.shape)




# import torch
# from torch import nn
# # 参考https://blog.csdn.net/cp1314971/article/details/104417808
#
# def conv_block(in_channel, out_channel):
#     layer = nn.Sequential(
#         nn.BatchNorm2d(in_channel),
#         nn.ReLU(),
#         nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
#     )
#     return layer
#
# class dense_block(nn.Module):
#     def __init__(self, in_channel, growth_rate, num_layers):
#         super(dense_block, self).__init__()
#         block = []
#         channel = in_channel
#         for i in range(num_layers):
#             block.append(conv_block(channel, growth_rate))
#             channel += growth_rate
#         self.net = nn.Sequential(*block)
#
#     def forward(self, x):
#         for layer in self.net:
#             out = layer(x)
#             x = torch.cat((out, x), dim=1)
#         return x
#
# def transition_down(in_channel, out_channel):
#     trans_layer = nn.Sequential(
#         nn.Conv2d(in_channel, out_channel, 3, padding=1),
#         nn.AvgPool2d(2, 2),
#         nn.BatchNorm2d(out_channel),
#         nn.ReLU()
#     )
#     return trans_layer
#
# def transition_up(in_channel, out_channel):
#     trans_layer = nn.Sequential(
#         nn.Conv2d(in_channel, out_channel, 1),
#         nn.Upsample(scale_factor=2, mode='bilinear'),
#         nn.BatchNorm2d(out_channel),
#         nn.ReLU()
#     )
#     return trans_layer
# class densenet(nn.Module):
#     def __init__(self, in_channel, num_classes, growth_rate=16, block_layers=[2,4]):
#         super(densenet, self).__init__()
#         self.first_block = nn.Sequential(
#             nn.Conv2d(in_channel, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True)
#             )
#         self.DB1 = self._make_dense_block(16, growth_rate,num=block_layers[0])
#         self.TL1 = self._make_transition_layer_down(48)
#         self.DB2 = self._make_dense_block(48, growth_rate, num=block_layers[0])
#         self.TL2 = self._make_transition_layer_down(80)
#         self.DB3 = self._make_dense_block(80, growth_rate, num=block_layers[0])
#         self.TL3 = self._make_transition_layer_down(112)
#         self.DB4 = self._make_dense_block(112, growth_rate, num=block_layers[0])
#         self.TL4 = self._make_transition_layer_down(144)
#
#         self.DB5 = self._make_dense_block(144, growth_rate, num=block_layers[1])
#         self.TL5 = self._make_transition_layer_up(208, 112)
#         self.DB6 = self._make_dense_block(112, growth_rate, num=block_layers[1])
#         self.TL6 = self._make_transition_layer_up(176, 80)
#         self.DB7 = self._make_dense_block(80, growth_rate, num=block_layers[1])
#         self.TL7 = self._make_transition_layer_up(144, 48)
#         self.DB8 = self._make_dense_block(48, growth_rate, num=block_layers[1])
#         self.TL8 = self._make_transition_layer_up(112, 112)
#
#     def forward(self, x):
#         x = self.first_block(x)
#         x = self.DB1(x)
#         x = self.TL1(x)
#         x = self.DB2(x)
#         x = self.TL2(x)
#         x = self.DB3(x)
#         x = self.TL3(x)
#         x = self.DB4(x)
#         x = self.TL4(x)
#         x = self.DB5(x)
#         x = self.TL5(x)
#         x = self.DB6(x)
#         x = self.TL6(x)
#         x = self.DB7(x)
#         x = self.TL7(x)
#         x = self.DB8(x)
#         x = self.TL8(x)
#         return x
#
#     def _make_dense_block(self,channels, growth_rate, num):
#         block = []
#         block.append(dense_block(channels, growth_rate, num))
#         channels += num * growth_rate
#
#         return nn.Sequential(*block)
#     def _make_transition_layer_down(self,channels):
#         block = []
#         block.append(transition_down(channels, channels))
#         return nn.Sequential(*block)
#
#     def _make_transition_layer_up(self, in_channels,out_channels):
#         block = []
#         block.append(transition_up(in_channels, out_channels))
#         return nn.Sequential(*block)
#
#
# net = densenet(3,10)
# x = torch.rand(1,3,512,512)
# for name,layer in net.named_children():
#     if name != "classifier":
#         x = layer(x)
#         print(name, 'output shape:', x.shape)
#     else:
#         x = x.view(x.size(0), -1)
#         x = layer(x)
#         print(name, 'output shape:', x.shape)


