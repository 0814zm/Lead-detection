import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from pandas import np

from utils.tools import initialize_weights
from models.resnet import BasicBlock as ResBlock
from models.GSConv import GatedSpatialConv2d
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from functools import partial
import numpy as np
from einops import rearrange

# -------------------------------------------------#
#   Basic Convolution Block
#   Conv2d + BatchNorm2d + LeakyReLU
# -------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class PatchEmbed(nn.Module):
    def __init__(self, input_shape, in_chans, patch_size=1,  num_features=128, norm_layer=None,
                 flatten=True):
        super().__init__()
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=1, stride=1)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, 64, 32, 32 -> B, 64, 16, 16
        # print("first x shape", x.shape)
        x = self.proj(x)

        # B, 64, 16, 16 -> B, 64, 256 -> B, 256, 64
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        # print("x shape to vit", x.shape)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# --------------------------------------------------------------------------------------------------------------------#
#   Global Attention
#   divide the feature into q, k, v
# --------------------------------------------------------------------------------------------------------------------#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GlobalAttention(nn.Module):
    def __init__(self, channels, input_shape, patch_size=1, depth=1, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.05,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU):
        super().__init__()

        self.patch_embed = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=channels,
                                      num_features=channels)
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, channels))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                ViTBlock(
                    dim=channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )

        self.norm = norm_layer(channels)

    def forward_features(self, x):
        x = self.patch_embed(x)

        # img_token_pe = x

        # -> batch, channel, height, width
        img_token_pe = self.pos_embed.view(1, *self.feature_shape, -1).permute(0, 3, 1, 2)

        # batch, height, width, channel -> B, N, C
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)
        # print("x shape:", x.shape)
        # print("img_token_pe shape:", img_token_pe.shape)

        x = self.pos_drop(x + img_token_pe)
        x = self.blocks(x)
        x = self.norm(x)
        x = rearrange(x, 'b (w h) c -> b c w h', w=self.feature_shape[0])


        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class ma_block(nn.Module):
    def __init__(self, channels, input_shape):
        super().__init__()

        self.conv_attention = cbam_block(channel=channels, kernel_size=3, ratio=8)
        self.vit_attention = GlobalAttention(channels=channels, input_shape=input_shape, patch_size=1)

        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        vit_x = x

        x = self.conv_attention(x)
        vit_x = self.vit_attention(vit_x)
        # print("conv_attention shape:", x.shape)
        # print("vit_attention shape:", vit_x.shape)

        x = x + vit_x

        x = self.norm(x)
        x = self.activation(x)

        return x


class SE_Block(nn.Module): # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out
class SE_Block1(nn.Module): # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block1, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out

class LAU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LAU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        out = out7 + out5 + out3
        out = out * out1
        out = self.conv1x1(out)

        return out


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 获取查询、键和值:通过线性变换将输入序列映射为查询、键和值序列
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # 计算相似度得分:对于每个查询，计算其与所有键之间的相似度。可以使用点积或加性注意力来计算相似度。
        sim_map = torch.bmm(query, key)  # 用于执行批次矩阵乘法（batch matrix multiplication）操作,输入是三维张量，并执行批次中的矩阵乘法运算:（batch_size, n, m）*（batch_size, m, p）=（batch_size, n, p）
        sim_map = F.softmax(sim_map, dim=-1)

        # 加权求和
        attention = torch.bmm(value, sim_map.permute(0, 2, 1))
        attention = attention.view(batch_size, channels, height, width)

        # 融合注意力::将归一化的注意力权重与值序列相乘，并对结果进行加权求和，得到位置注意力的输出。
        out = self.gamma * attention + x
        return out

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out:',avg_out.shape)
        a=torch.cat([max_out, avg_out], dim=1)
        # print('a:',a.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print('spatial:',spatial_out.shape)
        x = spatial_out * x
        # print('x:',x.shape)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class ASPP_1(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP_1, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=16 * rate, dilation=16 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch6_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch6_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch6_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 6, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共六个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        conv3x3_4 = self.branch5(x)  # [1, 256, 32, 32]
        # -----------------------------------------#
        #   第六个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch6_conv(global_feature)
        global_feature = self.branch6_bn(global_feature)
        global_feature = self.branch6_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将六个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature, result


def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model

def ResNet50(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained)
    return model

class CoordinateAttentionModule(nn.Module):
    def __init__(self, in_channels, height, width):
        super(CoordinateAttentionModule, self).__init__()

        self.coord_conv = nn.Conv2d(2, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.height = height
        self.width = width

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # 创建坐标矩阵
        coord_map = self.create_coordinate_map(batch_size)

        # 使用卷积操作将坐标矩阵映射为特征图
        coord_map = self.coord_conv(coord_map.cuda())  # [1, 1280, 32, 32]

        # 计算相似度得分
        sim_map = torch.matmul(coord_map.view(batch_size, channels, -1), x.view(batch_size, channels, -1).permute(0, 2, 1))
        sim_map = F.softmax(sim_map, dim=-1)  # [1, 1280, 1280]

        # 加权求和:前面：[1, 1024, 1280]，后：[1, 1280, 1280]
        attention = torch.matmul(x.view(batch_size, channels, -1).permute(0, 2, 1), sim_map)
        attention = attention.view(batch_size, channels, self.height, self.width)

        # 融合注意力
        out = self.gamma * attention + x
        return out

    def create_coordinate_map(self, batch_size):
        coord_map = torch.zeros((batch_size, 2, self.height, self.width), dtype=torch.float32)
        for i in range(self.height):
            coord_map[:, 0, i, :] = i / (self.height - 1)
        for j in range(self.width):
            coord_map[:, 1, :, j] = j / (self.width - 1)
        return coord_map


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        out = self.final(x)

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_PAM_SA1_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_PAM_SA1_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_SA(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SA, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 128, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 64, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 64, 512, 512]
        ss = self.res2(ss)  # [1, 64, 512, 512]
        ss = self.d2(ss)  # [1, 32, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 32, 512, 512]
        ss = self.res3(ss)  # [1, 32, 512, 512]
        ss = self.d3(ss)  # [1, 16, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 16, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out,g1,g2 ,g3

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA_PAM_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA_PAM_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 64, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 64, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 64, 512, 512]
        ss = self.res2(ss)  # [1, 64, 512, 512]
        ss = self.d2(ss)  # [1, 32, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 32, 512, 512]
        ss = self.res3(ss)  # [1, 32, 512, 512]
        ss = self.d3(ss)  # [1, 16, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 16, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_SE_LAU_allin_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_LAU_allin_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()


        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1328)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1328, 1328, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1328),
            nn.ReLU(inplace=True),
        )

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        low_level_features = self.lau(low_level_features)  # [1, 128, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        low_level_features = F.interpolate(low_level_features, size=(int(math.ceil(low_level_features.size()[-2] / 4)),
                                   int(math.ceil(low_level_features.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 48, 32, 32]

        x_and_low = torch.cat((x, low_level_features), dim=1)  # [1, 1328, 32, 32]

        se_aspp = self.senet(x_and_low)  # [1, 1328, 1, 1]
        se_feature_cat = se_aspp * x_and_low  # [1, 1328, 32, 32]
        x_and_low = self.conv_cat(se_feature_cat)  # [1, 1328, 32, 32]

        x = x_and_low[:, :1280, :, :]
        low_level_features = x_and_low[:, 1280:, :, :]
        low_level_features = F.interpolate(low_level_features, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 48, 128, 128]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]




        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_SE_LAU_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_LAU_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        se_aspp = self.senet(x)  # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_LAU_SE_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_LAU_SE_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.senet = SE_Block(in_planes=304)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(304, 304, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(304),
            nn.ReLU(inplace=True),
        )

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]

        se_aspp = self.senet(x)  # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)

        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_PAM_LAU_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_PAM_LAU_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.pam = PAM(1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_LAU_res50_ASPPr(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True, downsample_factor=16):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_LAU_res50_ASPPr, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        self.aspp = ASPP_1(dim_in=2048, dim_out=256, rate=16 // downsample_factor)

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_feature, x = self.aspp(x)  # x:[1, 256, 32, 32]

        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]


        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_SE_PAM_SA1_LAU_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_PAM_SA1_LAU_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)  # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_LAU_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_LAU_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_SEPAM_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SEPAM_res50, self).__init__()
        self.num_classes = num_classes

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SEPAM_GAU_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SEPAM_GAU_res50, self).__init__()
        self.num_classes = num_classes

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.global_avg_pool1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(256, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=5, padding=15)  # o = s(i-1) - 2p + k

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)

        x_1 = self.deconv(x)  # [1, 256, 128, 128]
        x_2 = self.global_avg_pool1(x)  # [1, 256, 1, 1]
        x_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_2 = self.deconv(x_2)  # [1, 256, 128, 128]

        x = torch.add(x_1, x_2)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_SE_PAM_SA1_LAU_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_PAM_SA1_LAU_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)  # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1SE(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1SE, self).__init__()
        self.num_classes = n_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

        self.d0 = nn.Conv2d(256, 1, kernel_size=1)
        self.res1 = ResBlock(1, 1)
        self.d1 = nn.Conv2d(1, 1, kernel_size=1)
        self.res2 = ResBlock(1, 1)
        self.d2 = nn.Conv2d(1, 1, kernel_size=1)
        self.res3 = ResBlock(1, 1)
        self.d3 = nn.Conv2d(1, 1, kernel_size=1)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(1, 1)
        self.gate2 = GatedSpatialConv2d(1, 1)
        self.gate3 = GatedSpatialConv2d(1, 1)

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet1 = SE_Block1(in_planes=3)
        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss1, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss2 = self.res2(ss1)  # [1, 1, 512, 512]
        ss2 = self.d2(ss2)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss2, g2 = self.gate2(ss2, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss3 = self.res3(ss2)  # [1, 1, 512, 512]
        ss3 = self.d3(ss3)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss3, g3 = self.gate3(ss3, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]

        ss = torch.cat((ss1, ss2, ss3), dim=1)  # [1, 3, 512, 512]
        se_ss = self.senet1(ss)  # [1, 3, 1, 1]
        se_feature_cat = se_ss * ss  # [1, 3, 512, 512]
        ss = self.conv_cat1(se_feature_cat) # [1, 3, 512, 512]

        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SA1(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SA1, self).__init__()
        self.num_classes = n_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SA1_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SA1_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1SE(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1SE, self).__init__()
        self.num_classes = n_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

        self.d0 = nn.Conv2d(256, 1, kernel_size=1)
        self.res1 = ResBlock(1, 1)
        self.d1 = nn.Conv2d(1, 1, kernel_size=1)
        self.res2 = ResBlock(1, 1)
        self.d2 = nn.Conv2d(1, 1, kernel_size=1)
        self.res3 = ResBlock(1, 1)
        self.d3 = nn.Conv2d(1, 1, kernel_size=1)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(1, 1)
        self.gate2 = GatedSpatialConv2d(1, 1)
        self.gate3 = GatedSpatialConv2d(1, 1)

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet1 = SE_Block1(in_planes=3)
        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss1, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss2 = self.res2(ss1)  # [1, 1, 512, 512]
        ss2 = self.d2(ss2)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss2, g2 = self.gate2(ss2, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss3 = self.res3(ss2)  # [1, 1, 512, 512]
        ss3 = self.d3(ss3)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss3, g3 = self.gate3(ss3, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]

        ss = torch.cat((ss1, ss2, ss3), dim=1)  # [1, 3, 512, 512]
        se_ss = self.senet1(ss)  # [1, 3, 1, 1]
        se_feature_cat = se_ss * ss  # [1, 3, 512, 512]
        ss = self.conv_cat1(se_feature_cat) # [1, 3, 512, 512]

        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SA1SE(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SA1SE, self).__init__()
        self.num_classes = n_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

        self.d0 = nn.Conv2d(256, 1, kernel_size=1)
        self.res1 = ResBlock(1, 1)
        self.d1 = nn.Conv2d(1, 1, kernel_size=1)
        self.res2 = ResBlock(1, 1)
        self.d2 = nn.Conv2d(1, 1, kernel_size=1)
        self.res3 = ResBlock(1, 1)
        self.d3 = nn.Conv2d(1, 1, kernel_size=1)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(1, 1)
        self.gate2 = GatedSpatialConv2d(1, 1)
        self.gate3 = GatedSpatialConv2d(1, 1)

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block1(in_planes=3)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss1, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss2 = self.res2(ss1)  # [1, 1, 512, 512]
        ss2 = self.d2(ss2)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss2, g2 = self.gate2(ss2, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss3 = self.res3(ss2)  # [1, 1, 512, 512]
        ss3 = self.d3(ss3)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss3, g3 = self.gate3(ss3, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]

        ss = torch.cat((ss1, ss2, ss3), dim=1)  # [1, 3, 512, 512]
        se_ss = self.senet(ss)  # [1, 3, 1, 1]
        se_feature_cat = se_ss * ss  # [1, 3, 512, 512]
        ss = self.conv_cat(se_feature_cat) # [1, 3, 512, 512]

        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)  # [1, 1280, 32, 32]


        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1_PAM_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_PAM_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_Mix_SA1_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_Mix_SA1_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.mix = ma_block(1280, [32, 32])
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        mix_attention = self.mix(x) # [1, 1280, 32, 32]
        x = self.conv_cat(mix_attention)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1_PAM_res50_hh(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_PAM_res50_hh, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(1, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input = input[:, :1, :, :]
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1_PAM_res50_hv(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_PAM_res50_hv, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(1, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input = input[:, 1:2, :, :]
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1_PAM_res50_4(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_PAM_res50_4, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(4, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_res50_SE_PAM_SA1cDecoder(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_res50_SE_PAM_SA1cDecoder, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self.final_conv2 = nn.Conv2d(2, 1, 1, bias=False)
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]


        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        x = torch.cat((x, ss), dim=1)  # [1, 2, 512, 512]
        x = self.final_conv2(x)

        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA_PAM_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA_PAM_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

         # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_PAM_SE_SA1_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_PAM_SE_SA1_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)


        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_SE_SA1_CAM_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_CAM_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.cam = CoordinateAttentionModule(1280,32,32)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.cam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SECPAM_SA1_RES101(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SECPAM_SA1_RES101, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )
        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)  # [1, 1280, 32, 32]
        x = self.pam(x)
        x = self.conv_cat(x)


        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)
class DeepLabv3_plus_SE_SA1(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)  # [1, 1280, 32, 32]


        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_res50_SE_PAM_SA1_edgeCdecoder(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_res50_SE_PAM_SA1_edgeCdecoder, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)
        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.expand = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(inplace=True))

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(305, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)  # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        ### Canny Edge
        im_arr = np.mean(input.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((input_size[0], 1, input_size[2], input_size[3]))
        for i in range(input_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 100, 200, 5)
        canny = torch.from_numpy(canny).cuda().float()
        ### End Canny Edge

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        cat = torch.cat((ss, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        edge = self.expand(acts)  # [1, 1, 512, 512]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        # 使用下采样方法对feat_map1进行尺寸调整
        downsampled_edge = F.interpolate(edge, size=(128, 128), mode='bilinear', align_corners=False)  # [1, 1, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features, downsampled_edge), dim=1)  # [1, 305, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1cEdge_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1cEdge_res50, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.finalconv = nn.Conv2d(2, 1, 1, bias=False)
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)  # [1, 1280, 32, 32]


        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        out = self.finalconv(torch.cat((x, ss), dim=1))

        seg_out = self.final(out)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_res50, self).__init__()
        self.num_classes = num_classes


        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]
        # print('x :{}'.format(x.shape))

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        out = self.final(x)

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1_PAM_res50_2(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_PAM_res50_2, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(2, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input = input[:, :2, :, :]
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x) # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_SE_SA1_2(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE_SA1_2, self).__init__()
        self.num_classes = n_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(2, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        input = input[:, :2, :, :]
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)  # [1, 1280, 32, 32]


        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_PAM_SE_SA1_res50_sdf(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_PAM_SE_SA1_res50_sdf, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)  # [1, 1280, 32, 32]

        se_aspp = self.senet(x)  # [1, 1280, 1, 1]
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)
        x1 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        sdf_out = self.tanh(x1)

        return seg_out, edge_out, sdf_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# 原文链接：https://blog.csdn.net/qq_45014374/article/details/127782301
class DeepLabv3_plus_CBAM(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_CBAM, self).__init__()
        self.num_classes = n_classes

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.cbam=CBAMLayer(channel=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        cbam_aspp = self.cbam(x)
        x = self.conv_cat(cbam_aspp)  # [1, 1280, 32, 32]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_CBAM_SA1(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_CBAM_SA1, self).__init__()
        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.cbam=CBAMLayer(channel=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        cbam_aspp = self.cbam(x)
        x = self.conv_cat(cbam_aspp)  # [1, 1280, 32, 32]


        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# 参考链接：https://blog.csdn.net/qq_45014374/article/details/127507120
class DeepLabv3_plus_SE(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_SE, self).__init__()
        self.num_classes = n_classes

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.senet = SE_Block(in_planes=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        se_aspp = self.senet(x)
        se_feature_cat = se_aspp * x  # [1, 1280, 32, 32]
        x = self.conv_cat(se_feature_cat)  # [1, 1280, 32, 32]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_CBAM(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_CBAM, self).__init__()
        self.num_classes = n_classes

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)  # 选用ResNet-101残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.cbam=CBAMLayer(channel=1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        cbam_aspp = self.cbam(x)
        x = self.conv_cat(cbam_aspp)  # [1, 1280, 32, 32]

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_2_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_2_res50, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet50(2, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        # print('input:', input.shape)
        input = input[:, :2, :, :]  # [1, 1, 512, 512]
        # print('input:', input.shape)
        x, low_level_features = self.resnet_features(input)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        out = self.final(x)

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_2(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_2, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(2, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        # print('input:', input.shape)
        input = input[:, :2, :, :]  # [1, 1, 512, 512]
        # print('input:', input.shape)
        x, low_level_features = self.resnet_features(input)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        out = self.final(x)

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus_sdf(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_sdf, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        self.tanh = nn.Tanh()

        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        out_seg = self.final(x)
        x1 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        out_tanh = self.tanh(x1)

        return out_tanh, out_seg

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_PAM_SA1_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_PAM_SA1_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.pam = PAM(1280)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_PAM_SA1_LAU_res50(nn.Module):
    def __init__(self, num_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_PAM_SA1_LAU_res50, self).__init__()
        self.num_classes = num_classes
        print("w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet50(3, os, pretrained=pretrained)  # 选用ResNet-50残差网络作为主干特征提取网络

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.pam = PAM(1280)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(1280, 1280, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.lau = LAU(256, 128)

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(
            input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]
        x = self.pam(x)  # [1, 1280, 32, 32]
        x = self.conv_cat(x)

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3 0-1 [1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)

        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.lau(low_level_features)

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:128——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        seg_out = self.final(x)

        return seg_out, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class DeepLabv3_plus_sdf_SA1(nn.Module):
    def __init__(self, n_classes=1, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
        super(DeepLabv3_plus_sdf_SA1, self).__init__()
        self.num_classes = n_classes
        print("SAUNet w/ Shape Stream")
        self.sigmoid = nn.Sigmoid()

        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(256, 1, kernel_size=1)

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

        # Atrous Conv
        self.resnet_features = ResNet101(3, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.tanh = nn.Tanh()
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, input):
        input_size = input.size()
        x, low_level_features = self.resnet_features(input)  # x[1, 2048, 32, 32]  low_level_features:torch.Size([1, 256, 128, 128])

        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # [1, 256, 32, 32]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 1280, 32, 32]

        # Shape Stream
        ss = F.interpolate(self.d0(x1), input_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)  # [1, 1, 512, 512]
        c3 = F.interpolate(self.c3(x2), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]，c3的作用，256——>1
        ss = self.d1(ss)  # [1, 1, 512, 512]
        ss, g1 = self.gate1(ss, c3)  # g1[1, 1, 512, 512]  ss [1, 1, 512, 512]
        ss = self.res2(ss)  # [1, 1, 512, 512]
        ss = self.d2(ss)  # [1, 1, 512, 512]
        c4 = F.interpolate(self.c4(x3), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g2 = self.gate2(ss, c4)  # g2 [1, 1, 512, 512] ss [1, 1, 512, 512]
        ss = self.res3(ss)  # [1, 1, 512, 512]
        ss = self.d3(ss)  # [1, 1, 512, 512]
        c5 = F.interpolate(self.c5(x4), input_size[2:],
                           mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        ss, g3 = self.gate3(ss, c5)  # g3[1, 1, 512, 512] ss[1, 1, 512, 512]
        ss = self.fuse(ss)  # [1, 1, 512, 512]
        ss = F.interpolate(ss, input_size[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        edge_out = self.sigmoid(ss)



        x = self.conv1(x)  # [1, 256, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)  # [1, 256, 128, 128]

        low_level_features = self.conv2(low_level_features)  # [1, 48, 128, 128],conv2:256——>48
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)  # [1, 304, 128, 128]
        x = self.last_conv(x)  # [1, 1, 128, 128]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 512, 512]
        out_seg = self.final(x)
        x1 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        out_tanh = self.tanh(x1)

        return out_seg, out_tanh, edge_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus_SECPAM_SA1_RES101(1)
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        seg_out, edge_out = model(image)
        # seg_out = model(image)
    print('seg out:', seg_out.shape)
    # print('edge out:', edge_out.shape)
