import torch.nn as nn
import torch


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nf + 0 * gc, gc, (3,3), padding=1, bias=False), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(nf + 1 * gc, gc, (3,3), padding=1, bias=False), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(nf + 2 * gc, gc, (3,3), padding=1, bias=False), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(nf + 3 * gc, gc, (3,3), padding=1, bias=False), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, (3,3), padding=1, bias=False), nn.ReLU())

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(nf, gc)
        self.layer2 = ResidualDenseBlock(nf, gc)
        self.layer3 = ResidualDenseBlock(nf, gc)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(nf, scale_factor=3):
    block = []
    for _ in range(scale_factor//2):
        block += [
            nn.Conv2d(nf, nf * (scale_factor ** 2), 1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.ReLU()
        ]
    return nn.Sequential(*block)


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class ResidualInResidualDenseNetwork(nn.Module):
    def __init__(self, nf,gc,n_basic_block=23):
        super(ResidualInResidualDenseNetwork, self).__init__()
        # for _ in range(n_basic_block):

        if block_num == 23:
            self.layer1 = ResidualInResidualDenseBlock(nf, gc)
            self.layer2 = ResidualInResidualDenseBlock(nf, gc)
            self.layer3 = ResidualInResidualDenseBlock(nf, gc)
            self.layer4 = ResidualInResidualDenseBlock(nf, gc)
            self.layer5 = ResidualInResidualDenseBlock(nf, gc)
            self.layer6 = ResidualInResidualDenseBlock(nf, gc)
            self.layer7 = ResidualInResidualDenseBlock(nf, gc)
            self.layer8 = ResidualInResidualDenseBlock(nf, gc)
            self.layer9 = ResidualInResidualDenseBlock(nf, gc)
            self.layer10 = ResidualInResidualDenseBlock(nf, gc)
            self.layer11 = ResidualInResidualDenseBlock(nf, gc)
            self.layer12 = ResidualInResidualDenseBlock(nf, gc)
            self.layer13 = ResidualInResidualDenseBlock(nf, gc)
            self.layer14 = ResidualInResidualDenseBlock(nf, gc)
            self.layer15 = ResidualInResidualDenseBlock(nf, gc)
            self.layer16 = ResidualInResidualDenseBlock(nf, gc)
            self.layer17 = ResidualInResidualDenseBlock(nf, gc)
            self.layer18 = ResidualInResidualDenseBlock(nf, gc)
            self.layer19 = ResidualInResidualDenseBlock(nf, gc)
            self.layer20 = ResidualInResidualDenseBlock(nf, gc)
            self.layer21 = ResidualInResidualDenseBlock(nf, gc)
            self.layer22 = ResidualInResidualDenseBlock(nf, gc)
            self.layer23 = ResidualInResidualDenseBlock(nf, gc)

    def forward(self, x):
        out_1 = self.layer1(x)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        out_6 = self.layer6(out_5)
        out_7 = self.layer7(out_6)
        out_8 = self.layer8(out_7)
        out_9 = self.layer9(out_8)
        out_10 = self.layer10(out_9)
        out_11 = self.layer11(out_10)
        out_12 = self.layer12(out_11)
        out_13 = self.layer13(out_12)
        out_14 = self.layer14(out_13)
        out_15 = self.layer15(out_14)
        out_16 = self.layer16(out_15)
        out_17 = self.layer17(out_16)
        out_18 = self.layer18(out_17)
        out_19 = self.layer19(out_18)
        out_20 = self.layer20(out_19)
        out_21 = self.layer21(out_20)
        out_22 = self.layer22(out_21)
        out_23 = self.layer23(out_22)

        return torch.cat((out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8,out_9,
                          out_10,out_11,out_12,out_13,out_14,out_15,out_16,out_17,
                          out_18,out_19,out_20,out_21,out_22,out_23),dim=1)


