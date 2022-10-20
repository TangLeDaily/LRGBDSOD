import torch
import torch.nn as nn
from rootmodel.model_util import *
from rootmodel.model_base import *

class CenterCombinePoolLayer(nn.Module):
    def __init__(self, in_channel):
        super(CenterCombinePoolLayer, self).__init__()
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.avg_pool = nn.AvgPool2d(3, 2, 1)
        self.attn_ScaleConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.attnConv1 = nn.Conv2d(in_channel * 2, in_channel, 1)
        self.attn_lastConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x):
        attn_pre_1 = self.relu(self.attn_ScaleConv1(x))
        attn_max_1 = self.max_pool(attn_pre_1)
        attn_avg_1 = self.avg_pool(attn_pre_1)
        attn_sum_1 = self.relu(self.attnConv1(torch.cat((attn_max_1, attn_avg_1), dim=1)))
        attn_sum_1 = self.upsample(attn_sum_1)

        attn_sum_1_out = self.attn_lastConv1(attn_sum_1)
        attn = torch.sigmoid(attn_sum_1_out)

        out = x * attn + x
        return out

class CenterCombineAttention(nn.Module):
    def __init__(self, in_channel=64):
        super(CenterCombineAttention, self).__init__()
        # temporal attention (before fusion conv)

        self.rgb_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.depth_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.fuse_Conv = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)
        self.fuse_outConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.Up1_pre = PixUpBlock(in_channel)
        self.CCP1 = CenterCombinePoolLayer(in_channel)
        self.Up1_after = PixUpBlock(in_channel)
        self.Up2_pre = PixUpBlock(in_channel//4)
        self.CCP2 = CenterCombinePoolLayer(in_channel//4)
        self.Up2_after = PixUpBlock(in_channel//4)
        self.Up3_pre = PixUpBlock(in_channel//16)
        self.CCP3 = CenterCombinePoolLayer(in_channel//16)
        self.Up3_after = PixUpBlock(in_channel//16)

        self.ConvLast = BasicBlock(in_channel//64, in_channel//64)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, rgb, depth):
        b, c, h, w = rgb.size()
        # temporal attention
        embedding_rgb = self.rgb_firstConv(rgb.clone())
        embedding_depth = self.depth_firstConv(depth.clone())
        corr = self.sigmoid(torch.sum(embedding_rgb * embedding_depth, 1).unsqueeze(1))
        fuse_fea = self.relu(self.fuse_Conv(torch.cat((embedding_rgb, embedding_depth), dim=1)))
        fuse_fea = corr * fuse_fea
        fuse_out = self.lrelu(self.fuse_outConv(fuse_fea))
        # print(fuse_out.size())
        # C=512, H=16, W=16

        Up1_pre = self.Up1_pre(fuse_out)
        Up1_after = self.Up1_after(self.CCP1(fuse_out))

        Up2_pre = self.Up2_pre(Up1_pre)
        Up2_after = self.Up2_after(self.CCP2(Up1_after+Up1_pre))

        Up3_pre = self.Up3_pre(Up2_pre)
        Up3_after = self.Up3_after(self.CCP3(Up2_after+Up2_pre))

        out = self.ConvLast(Up3_pre+Up3_after)
        # C=8, H=128, W=128
        # print(out.size())
        # print("asfdafas")
        return out

class JointAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(JointAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_convFC = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // ratio, in_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())
        #
        self.sum_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())

    def forward(self, rgb, depth):
        rgb_fea = self.avg_convFC(self.avg_pool(rgb)+self.max_pool(rgb))
        depth_fea = self.max_convFC(self.max_pool(depth)+self.avg_pool(depth))
        sum_fea = rgb_fea + depth_fea
        sum_fea = self.sum_convFC(sum_fea)
        rgb_out = rgb * sum_fea + rgb
        depth_out = depth * sum_fea + depth
        return rgb_out, depth_out

class FusionAttention(nn.Module):
    def __init__(self, in_channel = 64):
        super(FusionAttention, self).__init__()
        # temporal attention (before fusion conv)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.avg_pool = nn.AvgPool2d(3, 2, 1)
        self.rgb_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.depth_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.fuse_Conv = nn.Conv2d(in_channel*2, in_channel,3,1,1)
        self.fuse_outConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.attn_ScaleConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.attnConv1 = nn.Conv2d(in_channel * 2, in_channel, 1)
        self.attn_ScaleConv2 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.attnConv2 = nn.Conv2d(in_channel * 4, in_channel*2, 1)
        self.attn_lastConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.attn_lastConv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.Up = nn.PixelShuffle(2)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, rgb, depth):
        b, c, h, w = rgb.size()
        # temporal attention
        embedding_rgb = self.rgb_firstConv(rgb.clone())
        embedding_depth = self.depth_firstConv(depth.clone())
        corr = torch.sum(embedding_rgb*embedding_depth, 1).unsqueeze(1)
        fuse_fea = self.lrelu(self.fuse_Conv(torch.cat((embedding_rgb, embedding_depth), dim=1)))
        fuse_fea = corr * fuse_fea
        fuse_out = self.relu(self.fuse_outConv(fuse_fea))

        attn_pre_1 = self.lrelu(self.attn_ScaleConv1(fuse_fea))
        attn_max_1 = self.max_pool(attn_pre_1)
        attn_avg_1 = self.avg_pool(attn_pre_1)
        attn_sum_1 = self.lrelu(self.attnConv1(torch.cat((attn_max_1, attn_avg_1), dim=1)))
        attn_sum_1 = self.upsample(attn_sum_1)


        attn_sum_1_out = self.attn_lastConv1(attn_sum_1)
        # attn_sum_2_up = self.attn_lastConv2(self.Up(attn_sum_2))
        attn = torch.sigmoid(attn_sum_1_out)

        out = fuse_out * attn + fuse_out
        return out
class FFMCell(nn.Module):
    def __init__(self, c):
        super(FFMCell, self).__init__()
        self.xConv_first = nn.Conv2d(c, c, 3, 1, 1)
        self.yConv_first = nn.Conv2d(c, c, 3, 1, 1)
        self.xyConv = nn.Conv2d(c, c, 3, 1, 1)
        self.yPConv = nn.Conv2d(c, c, 3, 1, 1)
        self.zConv_a1 = nn.Conv2d(c, c, 3, 1, 1)
        self.zConv_a2 = nn.Conv2d(c, c, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x_ = self.xConv_first(x)
        y_ = self.yConv_first(y)
        xy_ = self.xyConv(x_+y_)
        y_p = self.tanh(self.yPConv(xy_ * y + xy_))
        z_a = self.sigmoid(self.zConv_a1(x)+self.zConv_a2(y_p))
        z_b = z_a * y_p
        x = x*(1-z_a) + z_b
        return x

class FFM(nn.Module):
    def __init__(self, c, nums=3):
        super(FFM, self).__init__()
        self.nums = nums
        self.cell1 = FFMCell(c)
        self.cell2 = FFMCell(c)
    def forward(self, r, d):
        r_x = r
        r_y = d

        d_x = r
        d_y = d
        for i in range(self.nums):
            r_x = self.cell1(r_x, r_y)
            d_y = self.cell2(d_y, d_x)
        return r_x, d_y

class PixUpBlock(nn.Module):
    def __init__(self, in_channel):
        super(PixUpBlock, self).__init__()
        self.up = nn.PixelShuffle(2)
        self.conv = BasicBlock(in_channel//4, in_channel//4)
    def forward(self, x):
        out = self.conv(self.up(x))
        return out

class MSJCA(nn.Module):
    def __init__(self):
        super(MSJCA, self).__init__()
        self.rgb_con1 = ResNet34_1()
        self.rgb_con2 = ResNet34_2()
        self.rgb_con3 = ResNet34_3()
        self.rgb_con4 = ResNet34_4()
        self.rgb_con5 = ResNet34_5()

        self.dep_con1 = ResNet34_1()
        self.dep_con2 = ResNet34_2()
        self.dep_con3 = ResNet34_3()
        self.dep_con4 = ResNet34_4()
        self.dep_con5 = ResNet34_5()

        self.JA1 = FFM(64)
        self.JA2 = FFM(64)
        self.JA3 = FFM(128)
        self.JA4 = FFM(256)
        self.JA5 = FFM(512)

    def forward(self, rgb_input, depth_input):
        rgb_1 = self.rgb_con1(rgb_input)
        depth_1 = self.dep_con1(depth_input)
        rgb_1, depth_1 = self.JA1(rgb_1, depth_1)

        rgb_2 = self.rgb_con2(rgb_1)
        depth_2 = self.dep_con2(depth_1)
        rgb_2, depth_2 = self.JA2(rgb_2, depth_2)

        rgb_3 = self.rgb_con3(rgb_2)
        depth_3 = self.dep_con3(depth_2)
        rgb_3, depth_3 = self.JA3(rgb_3, depth_3)

        rgb_4 = self.rgb_con4(rgb_3)
        depth_4 = self.dep_con4(depth_3)
        rgb_4, depth_4 = self.JA4(rgb_4, depth_4)

        rgb_5 = self.rgb_con5(rgb_4)
        depth_5 = self.dep_con5(depth_4)
        rgb_5, depth_5 = self.JA5(rgb_5, depth_5)

        ## remember 3,4,5
        return rgb_3, rgb_4, rgb_5, depth_3, depth_4, depth_5

class Decoder_A(nn.Module):
    def __init__(self):
        super(Decoder_A, self).__init__()
        self.rgb_con3 = DeResNet34_3()
        self.rgb_con4 = DeResNet34_4()
        self.rgb_con5 = DeResNet34_5()

        self.depth_con3 = DeResNet34_3()
        self.depth_con4 = DeResNet34_4()
        self.depth_con5 = DeResNet34_5()

        self.rgbEn3 = RFB(128, 128)
        self.rgbEn4 = RFB(256, 256)
        self.rgbEn5 = RFB(512, 512)
        self.depthEn3 = RFB(128, 128)
        self.depthEn4 = RFB(256, 256)
        self.depthEn5 = RFB(512, 512)

    def forward(self, rgb_3, rgb_4, rgb_5, depth_3, depth_4, depth_5):
        rgb_conv5_out = self.rgb_con5(self.rgbEn5(rgb_5))
        rgb_conv4_out = self.rgb_con4(self.rgbEn4(rgb_conv5_out + rgb_4))
        rgb_conv3_out = self.rgb_con3(self.rgbEn3(rgb_conv4_out + rgb_3))

        depth_conv5_out = self.depth_con5(self.depthEn5(depth_5))
        depth_conv4_out = self.depth_con4(self.depthEn4(depth_conv5_out + depth_4))
        depth_conv3_out = self.depth_con3(self.depthEn3(depth_conv4_out + depth_3))

        # rgb: torch.Size([4, 64, 64, 64])
        # depth: torch.Size([4, 64, 128, 128])
        return rgb_conv3_out, depth_conv3_out

class Decoder_B(nn.Module):
    def __init__(self):
        super(Decoder_B, self).__init__()
        self.rgb_con3 = DeResNet34_3()
        self.rgb_con4 = DeResNet34_4()
        self.rgb_con5 = DeResNet34_5()


        self.rgbEn3 = RFB(128, 128)
        self.rgbEn4 = RFB(256, 256)
        self.rgbEn5 = RFB(512, 512)

    def forward(self, rgb_3, rgb_4, rgb_5):
        rgb_conv5_out = self.rgb_con5(self.rgbEn5(rgb_5))
        rgb_conv4_out = self.rgb_con4(self.rgbEn4(rgb_conv5_out + rgb_4))
        rgb_conv3_out = self.rgb_con3(self.rgbEn3(rgb_conv4_out + rgb_3))

        # rgb: torch.Size([4, 64, 64, 64])
        # depth: torch.Size([4, 64, 128, 128])
        return rgb_conv3_out

# 可能的总体模型
class LRGBDSOD(nn.Module):
    def __init__(self, rfb_out_channel=32):
        super(LRGBDSOD, self).__init__()
        self.MSJCA = MSJCA()
        self.Decoder_B = Decoder_B()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.Decoder_A = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 3, 1, 1)
        )

        self.fattn1 = FusionAttention(128)
        self.fattn2 = FusionAttention(256)
        self.fattn3 = FusionAttention(512)

    def forward(self, low_input, high_input):
        # VGG
        rgb_1, rgb_2, rgb_3, depth_1, depth_2, depth_3 = self.MSJCA(low_input, high_input)

        fa_1 = self.fattn1(rgb_1, depth_1) # 128, 128, 64
        fa_2 = self.fattn2(rgb_2, depth_2) # 128, 128, 32
        fa_3 = self.fattn3(rgb_3, depth_3) # 64, 64 ,32

        # print(fa_1.size())
        # print(fa_2.size())
        # print(fa_3.size())
        # torch.Size([8, 128, 32, 32])
        # torch.Size([8, 256, 16, 16])
        # torch.Size([8, 512, 8, 8])
        out_Decode = self.Decoder_B(fa_1, fa_2, fa_3)
        # print(out_Decode.size())
        # torch.Size([8, 64, 64, 64])
        out = self.Decoder_A(out_Decode)

        return out

if __name__ == "__main__":
    a = torch.randn(8, 3, 256, 256)
    b = torch.randn(8, 3, 256, 256)
    model = LRGBDSOD()
    c = model(a, b)
    print(c.size())
