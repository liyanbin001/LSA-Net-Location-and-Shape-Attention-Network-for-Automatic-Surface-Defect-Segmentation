import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

__all__ = ['MPMAnet']
from torch.nn import Module, Parameter, Softmax

class VGGBlock_D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class VGGBlock_S(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class VGGBlock_R(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, (1, 3), stride=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, (1, 3), stride=(1, 1), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class VGGBlock_C(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class self_SP_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3_0_seg = VGGBlock_D(in_channels, in_channels, in_channels)
        self.pool =  nn.MaxPool2d(2, 2)
        self.down0 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down1 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down2 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down3 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down4 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down5 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down6 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down7 = nn.Conv2d(in_channels, in_channels//8, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.down8 = nn.Conv2d(in_channels // 8, 1, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, input):

        input  = self.conv3_0_seg(input)
        m_batchsize, C, height_input, width_input = input.size()
        self.pool_R = nn.MaxPool2d(kernel_size=(1, width_input), stride=(1, width_input), padding=(0, 0))
        self.pool_C = nn.MaxPool2d(kernel_size=(height_input, 1), stride=(height_input, 1), padding=(0, 0))
        out0 = self.down0(input)
        out0_R = self.pool_R(out0)
        out0_C = self.pool_C(out0)
        out0_R = F.interpolate(out0_R, (height_input, width_input), )
        out0_C = F.interpolate(out0_C, (height_input, width_input), )
        out0 = out0_R+out0_C
        out1 = self.down1(input)
        out1_R = self.pool_R(out1)
        out1_C = self.pool_C(out1)
        out1_R = F.interpolate(out1_R, (height_input, width_input), )
        out1_C = F.interpolate(out1_C, (height_input, width_input), )
        out1 = out1_R+out1_C
        out2 = self.down2(input)
        out2_R = self.pool_R(out2)
        out2_C = self.pool_C(out2)
        out2_R = F.interpolate(out2_R, (height_input, width_input), )
        out2_C = F.interpolate(out2_C, (height_input, width_input), )
        out2 = out2_R+out2_C
        out3 = self.down3(input)
        out3_R = self.pool_R(out3)
        out3_C = self.pool_C(out3)
        out3_R = F.interpolate(out3_R, (height_input, width_input), )
        out3_C = F.interpolate(out3_C, (height_input, width_input), )
        out3 = out3_R+out3_C
        out4 = self.down4(input)
        out4_R = self.pool_R(out4)
        out4_C = self.pool_C(out4)
        out4_R = F.interpolate(out4_R, (height_input, width_input), )
        out4_C = F.interpolate(out4_C, (height_input, width_input), )
        out4 = out4_R+out4_C
        out5 = self.down5(input)
        out5_R = self.pool_R(out5)
        out5_C = self.pool_C(out5)
        out5_R = F.interpolate(out5_R, (height_input, width_input), )
        out5_C = F.interpolate(out5_C, (height_input, width_input), )
        out5 = out5_R+out5_C
        out6 = self.down6(input)
        out6_R = self.pool_R(out6)
        out6_C = self.pool_C(out6)
        out6_R = F.interpolate(out6_R, (height_input, width_input), )
        out6_C = F.interpolate(out6_C, (height_input, width_input), )
        out6 = out6_R+out6_C
        out7 = self.down7(input)
        out7_R = self.pool_R(out7)
        out7_C = self.pool_C(out7)
        out7_R = F.interpolate(out7_R, (height_input, width_input), )
        out7_C = F.interpolate(out7_C, (height_input, width_input), )
        out7 = out7_R+out7_C
        out_cat = torch.cat([out0,out1,out2,out3,out4,out5,out6,out7],1)
        out_add = out0+out1+out2+out3+out4+out5+out6+out7
        out_add1 = self.down8(out_add)
        return out_cat,out_add1

class self_LSP_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3_0_seg = VGGBlock_R(in_channels, in_channels, in_channels)
        self.conv3_R_seg = VGGBlock_R(in_channels, in_channels, in_channels)
        self.conv3_C_seg = VGGBlock_C(in_channels, in_channels, in_channels)
        self.pool =  nn.MaxPool2d(2, 2)
        self.down1 = nn.Conv2d( in_channels, 1, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
    def forward(self, input):
        m_batchsize, C, height_input, width_input = input.size()
        #yuan
        self.pool_R = nn.MaxPool2d(kernel_size=(1, width_input//2), stride=(1, width_input//2), padding=(0, 0))
        self.pool_C = nn.MaxPool2d(kernel_size=(height_input//2, 1), stride=(height_input//2, 1), padding=(0, 0))
        #gai
        # self.pool_R = nn.MaxPool2d(kernel_size=(1, width_input), stride=(1, width_input), padding=(0, 0))
        # self.pool_C = nn.MaxPool2d(kernel_size=(height_input, 1), stride=(height_input, 1), padding=(0, 0))
        inputR = self.conv3_R_seg(input)
        outR = self.pool_R(inputR)
        outR = F.interpolate(outR, (height_input, width_input), )
        inputC = self.conv3_C_seg(input)
        outC = self.pool_C(inputC)
        outC = F.interpolate(outC, (height_input, width_input), )
        out = outR + outC
        Lout = self.conv3_0_seg(out)
        Lout1 = self.down1(Lout)
        return Lout,Lout1

class self_PSP_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3_0_seg = VGGBlock_D(in_channels, in_channels, in_channels)
        self.down0 = nn.Conv2d(2*in_channels, in_channels, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.down1 = nn.Conv2d( in_channels, 1, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
    def forward(self, input):
        m_batchsize, C, height_input, width_input = input.size()

        self.pool1 = nn.AvgPool2d(kernel_size=[5,5], stride=[5,5])
        self.pool2 = nn.AvgPool2d(kernel_size=[3,3], stride=[3,3])
        input = self.conv3_0_seg(input)
        out1 = self.pool1(input)
        out1 = F.interpolate(out1, (height_input, width_input), )
        out2 = self.pool2(input)
        out2 = F.interpolate(out2, (height_input, width_input), )
        out =  torch.cat([out1,out2],1)
        out = self.down0(out)
        Pout = self.conv3_0_seg(out)
        Pout1 = self.down1(Pout)
        return Pout,Pout1

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class LocSegnet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        nb_filter_0 = [32, 64, 128, 256, 512]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.con7x7 = nn.Conv2d(in_channels=3,out_channels=nb_filter_0[0],kernel_size=7,stride=1,padding=3)
        self.conv0_0_seg = VGGBlock_D(3, nb_filter_0[0], nb_filter_0[0])
        self.conv1_0_seg = VGGBlock_D(nb_filter_0[0], nb_filter_0[1], nb_filter_0[1])
        self.conv2_0_seg = VGGBlock_D(nb_filter_0[1], nb_filter_0[2], nb_filter_0[2])

        self.conv3_0_seg = VGGBlock_D(nb_filter_0[2], nb_filter_0[3], nb_filter_0[3])
        self.conv4_0_seg = VGGBlock_D(nb_filter_0[3], nb_filter_0[4], nb_filter_0[4])
        self.conv4_1_seg = VGGBlock_D(2*nb_filter_0[2] +2*nb_filter_0[3], nb_filter_0[3], nb_filter_0[3])
        self.SP_Block = self_SP_Block(nb_filter_0[0] + nb_filter_0[1])
        self.conv2_1_seg = VGGBlock_D(nb_filter_0[0] + nb_filter_0[1] , nb_filter_0[1], nb_filter_0[1])
        self.conv2_2_seg = VGGBlock_D(nb_filter_0[3] + nb_filter_0[2] , nb_filter_0[2], nb_filter_0[2])

        self.conv4_2_seg = VGGBlock_D(2*nb_filter_0[0] +2*nb_filter_0[1], nb_filter_0[0], nb_filter_0[0])

        self.LSP_Block = self_LSP_Block(nb_filter_0[2] + nb_filter_0[3])
        self.PSP_Block = self_PSP_Block(nb_filter_0[2] + nb_filter_0[3])
        self.conv3_1 = VGGBlock_S(nb_filter_0[3] + nb_filter_0[4], nb_filter_0[3], nb_filter_0[3])
        self.conv2_2 = VGGBlock_S(nb_filter_0[2] + nb_filter_0[3], nb_filter_0[2], nb_filter_0[2])
        self.SP_Block0 = self_SP_Block(nb_filter_0[2] + nb_filter_0[3])


        self.conv1_3 = VGGBlock_S(nb_filter_0[1] + nb_filter_0[2], nb_filter_0[1], nb_filter_0[1])
        self.conv0_4 = VGGBlock_S(nb_filter_0[0] + nb_filter_0[1], nb_filter_0[0], nb_filter_0[0])
        self.LSP_Block0 = self_LSP_Block(nb_filter_0[0] + nb_filter_0[1])
        self.PSP_Block0= self_PSP_Block(nb_filter_0[0] + nb_filter_0[1])

        self.final = nn.Conv2d(nb_filter_0[0], 1, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.conv13 = nn.Conv2d(3 * nb_filter_0[4], nb_filter_0[4], kernel_size=1)
        self.convd2 = nn.Conv2d(nb_filter_0[4],nb_filter_0[4],kernel_size=3,dilation=2,stride=1,padding=2)
        self.convd4 = nn.Conv2d(nb_filter_0[4], nb_filter_0[4], kernel_size=3, dilation=4, stride=1, padding=4)
    def forward(self, input,target, label):
        m_batchsize, C, height_input, width_input = input.size()
        x0_0 = self.conv0_0_seg(input)  # 3*256*256 --> 32*256*256
        x1_0 = self.pool(x0_0)          # 32*256*256 --> 32*128*128
        x1_0 = self.conv1_0_seg(x1_0)   # 32*128*128 --> 64*128*128
        x0_1 = self.pool(x0_0)
        out_cat,out_add1 = self.SP_Block(torch.cat([x0_1,x1_0],1))
        out_cat = self.conv2_1_seg(out_cat)
        label = F.interpolate(label, (height_input // 2, width_input // 2), )
        x2_0 = self.pool(x1_0 + out_cat)          # 64*128*128  --> 64*64*64
        x2_0 = self.conv2_0_seg(x2_0)  # 64*64*64--> 128*64*64
        x3_0 = self.pool(x2_0)         # 128*64*64 --> 128*32*32
        x3_0 = self.conv3_0_seg(x3_0)  # 128*32*32--> 256*32*32
        x3_1 = self.pool(x2_0)
        Lout, Lout1 = self.LSP_Block(torch.cat([x3_0, x3_1], 1))
        Pout, Pout1 = self.PSP_Block(torch.cat([x3_0, x3_1], 1))
        target = F.interpolate(target, (height_input//8, width_input//8), )
        x3_2 = self.conv4_1_seg(torch.cat([Lout, Pout], 1))

        x4_0 = self.pool(x3_0+x3_2)          # 256*32*32 --> 256*16*16
        x4_0 = self.conv4_0_seg(x4_0)  # 256*16*16 --> 512*16*16
        x4_0_2 = self.convd2(x4_0)
        x4_0_4 = self.convd4(x4_0)
        x4_1 = self.conv13(torch.cat([x4_0_4, x4_0_2, x4_0], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_1 )], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))  # cun zai out_cat zui jia wei 0.7391
        x1_3 = self.conv1_3(torch.cat([x1_0 , self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        output = self.final(x0_4)
        return output, Lout1, Pout1, out_add1, target, label, Lout, Pout

if __name__ == "__main__":
    input = torch.rand(5, 3, 256, 256)
    label_edge = torch.rand(5, 1, 256, 256)
    model = LocSegnet()
    output, Lout1, Pout1, out_add1, target, label, Lout, Pout = model(input,label_edge,label_edge)
