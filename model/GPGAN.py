from blocks.block import *
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from plots.plots_feature_img import show_save_lr_hr_img



class GPGAN_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, gc=32, scale_factor=3, n_basic_block=23):
        super(GPGAN_Generator, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, nf, (3,3),bias=False), nn.ReLU(negative_slope=0.2, inplace=True))

        self.basic_block = ResidualInResidualDenseNetwork(nf, gc, n_basic_block)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, (3,3),bias=False), nn.ReLU())
        self.upsample = upsample_block(nf, scale_factor=self.scale_factor)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, (3,3),bias=False), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, (3,3),bias=False), nn.ReLU())
        self.conv5 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, out_channels,(3,3),bias=False))
        
        self.upsample2 = nn.Upsample(scale_factor=self.scale_factor)
      
    def forward(self, x):
        
        x1 = self.conv1(x)
    
        x2 = self.basic_block(x1)
     
        x3 = self.upsample(self.conv2(x2))
        
        x4 = self.conv3(x3)
       
        x5 = self.conv4(x4)
 
        x6 = self.conv5(x5)
 
        return x6
        
class GPGAN_Generator_2(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, gc=32, scale_factor=3, n_basic_block=23):
        super(GPGAN_Generator, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, nf, (3,3),bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(nf, gc)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, (3,3),bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.upsample = upsample_block(nf, scale_factor=self.scale_factor)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, (3,3),bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, (3,3),bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, out_channels,(3,3),bias=False))
        
        self.upsample2 = nn.Upsample(scale_factor=self.scale_factor)
      
    def forward(self, x):
        
        x1 = self.conv1(x)
    
        x2, x_con = self.basic_block(x1)
     
        x3 = self.upsample(self.conv2(x2 + x_con))
        
        x4 = self.conv3(x3)
       
        x5 = self.conv4(x4)
 
        x6 = self.conv5(x5)
 
        return x6



class GPGAN_Discriminator(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)


    Arg:
        num_in_ch (int): Channel number of inputs. Default: 1.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(GPGAN_Discriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class Disc_U_Net(nn.Module):


    def __init__(self, num_in_ch, num_feat=64):
        super(Disc_U_Net, self).__init__()

        n1 = num_feat
        in_ch = num_in_ch
        out_ch = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels,num_feats,num_conv_block=4):
        super(Discriminator, self).__init__()

        block = []

        in_channels = in_channels
        out_channels = num_feats

        for _ in range(num_conv_block):
            block += [nn.ReflectionPad2d(2),
                      nn.Conv2d(in_channels, out_channels, 5,bias=False),
                      nn.InstanceNorm2d(out_channels),
                      nn.LeakyReLU(0.2),
                      ]
            in_channels = out_channels

            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3,bias=False),
                      nn.InstanceNorm2d(out_channels),
                      nn.LeakyReLU(0.2)]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels, out_channels, 3,bias=False),
                  nn.InstanceNorm2d(out_channels),
                  nn.LeakyReLU(0.2),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(out_channels, 1, 3,bias=False),
                  ]

        self.feature_extraction = nn.Sequential(*block)

        self.avgpool = nn.AdaptiveAvgPool2d((512, 512))

    def forward(self, x):
        x = self.feature_extraction(x)

        return x

class Discriminator2(nn.Module):
    def __init__(self, in_channels=1,num_feats=64,num_conv_block=4):
        super(Discriminator2, self).__init__()
        self.channels = in_channels
        self.height = 12 # 192 // 2 ** 4
        self.width = 12 # 192 // 2 ** 4
        self.num_feats = num_feats
        # Calculate output shape of image discriminator
        self.output_shape = (1, self.height , self.width)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(self.channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
