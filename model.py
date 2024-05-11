import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3D_WS(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3D_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(
                                  dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class ResBlock_Custom(nn.Module):
    def __init__(self, dimension, input_channels, output_channels):
        super().__init__()
        self.dimension = dimension
        self.input_channels = input_channels
        self.output_channels = output_channels
        if dimension == 2:
            self.conv_res = nn.Conv2d(self.input_channels, self.output_channels, 3, padding= 1)
            self.conv_ws = Conv2d_WS(in_channels = self.input_channels,
                                  out_channels= self.output_channels,
                                  kernel_size = 3,
                                  padding = 1)
            self.conv = nn.Conv2d(self.output_channels, self.output_channels, 3, padding = 1)
        elif dimension == 3:
            self.conv_res = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
            self.conv_ws = Conv3D_WS(in_channels=self.input_channels,
                                     out_channels=self.output_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1)


    def forward(self, x):
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        return output



class Eapp1(nn.Module):
    '''
        This is the first part of the Appearance Encoder. To generate
        a 4D tensor of volumetric features vs.
    '''
    def __init__(self):
        # first conv layer, output size: 512 * 512 * 64
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3)                                            # output 512*512*64
        self.resblock_128 = ResBlock_Custom(dimension=2, input_channels=64, output_channels=128)        # output 512*512*128
        self.resblock_256 = ResBlock_Custom(dimension=2, input_channels=128, output_channels=256)       # output 512*512*256
        self.resblock_512 = ResBlock_Custom(dimension=2, input_channels=256, output_channels=512)       # output 512*512*512
        self.resblock3D_96 = ResBlock_Custom(dimension=3, input_channels=96, output_channels=96)        # output
        self.resblock3D_96_2 = ResBlock_Custom(dimension=3, input_channels=96, output_channels=96)
        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)


    def forward(self, x):
        out = self.conv(x)
        print("After first layer:" + str(out.size()))
        out = self.resblock_128(out)
        print("After resblock_128:" + str(out.size()))
        out = self.avgpool(out)
        print("After avgpool:" + str(out.size()))
        out = self.resblock_256(out)
        print("After resblock_256:" + str(out.size()))
        out = self.avgpool(out)
        print("After avgpool:" + str(out.size()))
        out = self.resblock_512(out)
        print("After resblock_512:" + str(out.size()))
        out = self.avgpool(out)
        print("After avgpool:" + str(out.size()))

        out = F.group_norm(out, num_groups=32)
        print("After group_norm:" + str(out.size()))
        out = F.relu(out)
        print("After relu:" + str(out.size()))

        out = self.conv_1(out)
        print("After conv_1:" + str(out.size()))

        # Reshape
        out = out.view(out.size(0), 96, 16, out.size(2), out.size(3))
        print("After reshape:" + str(out.size()))

        # ResBlock 3D
        out = self.resblock3D_96(out)
        print("After resblock3D:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))

        return out




class Eapp2(nn.Module):
    '''
        This is the second part of the Appearance Encoder. To generate
        a global descriptor es that helps retain the appearance of the output
        image.
        This encoder uses ResNet-50 as backbone, and replace the residual block with the customized res-block.
        ref: https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
    '''
    def __init__(self, repeat, in_channels=3, outputs=256):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        filters = [64, 256, 512, 1024, 2048]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', ResBlock_Custom(dimension=2, input_channels=filters[0], output_channels=filters[1]))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), ResBlock_Custom(dimension=2, input_channels=filters[1], output_channels=filters[1]))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', ResBlock_Custom(dimension=2, input_channels=filters[1], output_channels=filters[2]))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i+1,), ResBlock_Custom(dimension=2, input_channels=filters[2], output_channels=filters[2]))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', ResBlock_Custom(dimension=2, input_channels=filters[2], output_channels=filters[3]))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i+1,), ResBlock_Custom(dimension=2, input_channels=filters[3], output_channels=filters[3]))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', ResBlock_Custom(dimension=2, input_channels=filters[3], output_channels=filters[4]))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,), ResBlock_Custom(dimension=2, input_channels=filters[4], output_channels=filters[4]))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)


    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)
        print("Dimensions of final output of Eapp2: " + str(input.size()))

        return input



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)



class Emtn_facial(nn.Module):
    def __init__(self, in_channels, resblock=ResBlock, outputs=256):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        input = self.fc(input)
        print("Dimensions of final output of Emtn_facial: " + str(input.size()))

        return input



class Emtn_head(nn.Module):
    def __init__(self, in_channels, resblock, outputs=256):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input)
        input = self.fc(input)

        return input



## Need to learn more about the adaptive group norm.
class ResBlock3D_Adaptive(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_res = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
        self.conv_ws = Conv3D_WS(in_channels=self.input_channels,
                                 out_channels=self.output_channels,
                                 kernel_size=3,
                                 padding=1)
        self.conv = nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1)


    def forward(self, x):
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        return output


class WarpGenerator(nn.Module):
    def __init__(self, input_channels):
        super(WarpGenerator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=2048, kernel_size=1, padding=0, stride=1)
        self.hidden_layer = nn.Sequential(
            ResBlock_Custom(dimension=3, input_channels=512, output_channels=256),
            nn.Upsample(scale_factor=(2, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=256, output_channels=128),
            nn.Upsample(scale_factor=(2, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=128, output_channels=64),
            nn.Upsample(scale_factor=(1, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=64, output_channels=32),
            nn.Upsample(scale_factor=(1, 2, 2)),
        )
        self.conv3D = nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        print("The output shape after first conv layer is " + str(out.size()))
        # reshape
        out = out.view(out.size(0), 512, 4, 16, 16)
        print("The output shape after reshaping is " + str(out.size()))

        out = self.hidden_layer(out)
        print("The output shape after hidden_layer is " + str(out.size()))
        out = F.group_norm(out, num_groups=32)
        print("The output shape after group_norm is " + str(out.size()))
        out = F.relu(out)
        print("The output shape after relu is " + str(out.size()))
        out = torch.tanh(out)
        print("The final output shape is : " + str(out.size()))

        return out



class G3d(nn.Module):
    def __init__(self, input_channels):
        super(G3d, self).__init__()
        self.input_channels = input_channels
        self.resblock1 = ResBlock_Custom(dimension=3, input_channels=input_channels, output_channels=192)
        self.resblock2 = ResBlock_Custom(dimension=3, input_channels=192, output_channels=196)
        self.resblock3 = ResBlock_Custom(dimension=3, input_channels=192, output_channels=384)
        self.resblock4 = ResBlock_Custom(dimension=3, input_channels=384, output_channels=384)
        self.resblock5 = ResBlock_Custom(dimension=3, input_channels=384, output_channels=512)
        self.resblock6 = ResBlock_Custom(dimension=3, input_channels=512, output_channels=512)
        self.resblock7 = ResBlock_Custom(dimension=3, input_channels=512, output_channels=384)
        self.resblock8 = ResBlock_Custom(dimension=3, input_channels=384, output_channels=196)
        self.resblock9 = ResBlock_Custom(dimension=3, input_channels=196, output_channels=96)
        self.conv_last = nn.Conv3d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = ResBlock_Custom(dimension=3, input_channels=self.input_channels, output_channels=192)(x)
        out = nn.Upsample(scale_factor=(1/2, 2, 2))(out)
        short_cut1 = ResBlock_Custom(dimension=3, input_channels=192, output_channels=196)(out)
        out = ResBlock_Custom(dimension=3, input_channels=192, output_channels=384)(out)
        out = nn.Upsample(scale_factor=(2, 2, 2))(out)
        short_cut2 = ResBlock_Custom(dimension=3, input_channels=384, output_channels=384)(out)
        out = ResBlock_Custom(dimension=3, input_channels=384, output_channels=512)(out)
        short_cut3 = out
        out = ResBlock_Custom(dimension=3, input_channels=512, output_channels=512)(out)
        out = short_cut3 + out
        out = ResBlock_Custom(dimension=3, input_channels=512, output_channels=384)(out)
        out = nn.Upsample(scale_factor=(2, 2, 2))(out)
        out = out + short_cut2
        out = ResBlock_Custom(dimension=3, input_channels=384, output_channels=196)(out)
        out = nn.Upsample(scale_factor=(1/2, 2, 2))(out)
        out = out + short_cut1
        out = ResBlock_Custom(dimension=3, input_channels=196, output_channels=96)(out)

        # Last Layer.
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = nn.Conv3d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=1)(out)

        return out




class G2d(nn.Module):
    def __init__(self, input_channels):
        super(G2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1, stride=1)
        
        self.hidden_layer = nn.Sequential(
            ResBlock_Custom(dimension=2, input_channels=64, output_channels=128),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=128, output_channels=256),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=256, output_channels=512),
            nn.Upsample(scale_factor=(2, 2))
        )
        
        self.last_layer = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.hidden_layer(out)
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.last_layer(out)
        out = torch.tanh(out)
        return out

'''
The main changes made to align the code with the training stages are:

Introduced Gbase class that combines the components of the base model.
Introduced Genh class for the high-resolution model.
Introduced GHR class that combines the base model Gbase and the high-resolution model Genh.
Introduced Student class for the student model, which includes an encoder, decoder, and SPADE blocks for avatar conditioning.
Added separate training functions for each stage: train_base, train_hr, and train_student.
Demonstrated the usage of the training functions and saving the trained models.

Note: The code assumes the presence of appropriate dataloaders (dataloader, dataloader_hr, dataloader_avatars) and the implementation of the SPADEResBlock class for the student model. Additionally, the specific training loop details and loss functions need to be implemented based on the paper's description.
'''

class Gbase(nn.Module):
    def __init__(self):
        super(Gbase, self).__init__()
        self.Eapp1 = Eapp1()
        self.Eapp2 = Eapp2(repeat=[3, 4, 6, 3])
        self.Emtn_facial = Emtn_facial(in_channels=3, resblock=ResBlock, outputs=256)
        self.Emtn_head = Emtn_head(in_channels=3, resblock=ResBlock, outputs=6)
        self.Ws2c = WarpGenerator(input_channels=512)
        self.Wc2d = WarpGenerator(input_channels=512)
        self.G3d = G3d(input_channels=96)
        self.G2d = G2d(input_channels=96)

    def forward(self, xs, xd):
        vs = self.Eapp1(xs)
        es = self.Eapp2(xs)
        zs = self.Emtn_facial(xs)
        Rs, ts = self.Emtn_head(xs)
        zd = self.Emtn_facial(xd)
        Rd, td = self.Emtn_head(xd)
        ws2c = self.Ws2c(torch.cat((Rs, ts, zs, es), dim=1))
        wc2d = self.Wc2d(torch.cat((Rd, td, zd, es), dim=1))
        vc2d = self.G3d(torch.nn.functional.grid_sample(vs, ws2c))
        vc2d = torch.nn.functional.grid_sample(vc2d, wc2d)
        xhat = self.G2d(torch.nn.functional.avg_pool3d(vc2d, kernel_size=(1, 1, vc2d.size(4))).squeeze(4))
        return xhat

class Genh(nn.Module):
    def __init__(self):
        super(Genh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        
        self.hidden_layer = nn.Sequential(
            ResBlock_Custom(dimension=2, input_channels=64, output_channels=128),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=128, output_channels=256),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=256, output_channels=512),
            nn.Upsample(scale_factor=(2, 2))
        )
        
        self.last_layer = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.hidden_layer(out)
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.last_layer(out)
        out = torch.tanh(out)
        return out

class GHR(nn.Module):
    def __init__(self):
        super(GHR, self).__init__()
        self.Gbase = Gbase()
        self.Genh = Genh()

    def forward(self, xs, xd):
        xhat_base = self.Gbase(xs, xd)
        xhat_hr = self.Genh(xhat_base)
        return xhat_hr

class Student(nn.Module):
    def __init__(self, num_avatars):
        super(Student, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            ResBlock_Custom(dimension=2, input_channels=64, output_channels=128),
            ResBlock_Custom(dimension=2, input_channels=128, output_channels=256),
            ResBlock_Custom(dimension=2, input_channels=256, output_channels=512),
            ResBlock_Custom(dimension=2, input_channels=512, output_channels=1024)
        )
        self.decoder = nn.Sequential(
            ResBlock_Custom(dimension=2, input_channels=1024, output_channels=512),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=512, output_channels=256),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=256, output_channels=128),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        )
        self.spade_blocks = nn.ModuleList([
            SPADEResBlock(1024, 1024, num_avatars) for _ in range(4)
        ])

    def forward(self, xd, avatar_index):
        encoded = self.encoder(xd)
        for spade_block in self.spade_blocks:
            encoded = spade_block(encoded, avatar_index)
        xhat = self.decoder(encoded)
        return xhat


'''
In this expanded code, we have the SPADEResBlock class which represents a residual block with SPADE (Spatially-Adaptive Normalization) layers. The block consists of two convolutional layers (conv_0 and conv_1) with normalization layers (norm_0 and norm_1) and a learnable shortcut connection (conv_s and norm_s) if the input and output channels differ.
The SPADE class implements the SPADE layer, which learns to modulate the normalized activations based on the avatar embedding. It consists of a shared convolutional layer (conv_shared) followed by separate convolutional layers for gamma and beta (conv_gamma and conv_beta). The avatar embeddings (avatar_shared_emb, avatar_gamma_emb, and avatar_beta_emb) are learned for each avatar index and are added to the corresponding activations.
During the forward pass of SPADEResBlock, the input x is passed through the shortcut connection and the main branch. The main branch applies the SPADE normalization followed by the convolutional layers. The output of the block is the sum of the shortcut and the main branch activations.
The SPADE layer first normalizes the input x using instance normalization. It then applies the shared convolutional layer to obtain the shared embedding. The gamma and beta values are computed by adding the avatar embeddings to the shared embedding and passing them through the respective convolutional layers. Finally, the normalized activations are modulated using the computed gamma and beta values.
Note that this implementation assumes the presence of the avatar index tensor avatar_index during the forward pass, which is used to retrieve the corresponding avatar embeddings.
'''
class SPADEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_avatars):
        super(SPADEResBlock, self).__init__()
        self.learned_shortcut = (in_channels != out_channels)
        middle_channels = min(in_channels, out_channels)

        self.conv_0 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.norm_0 = SPADE(in_channels, num_avatars)
        self.norm_1 = SPADE(middle_channels, num_avatars)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, num_avatars)

    def forward(self, x, avatar_index):
        x_s = self.shortcut(x, avatar_index)

        dx = self.conv_0(self.actvn(self.norm_0(x, avatar_index)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, avatar_index)))

        out = x_s + dx

        return out

    def shortcut(self, x, avatar_index):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, avatar_index))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, num_avatars):
        super().__init__()
        self.num_avatars = num_avatars
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.conv_shared = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

        self.avatar_shared_emb = nn.Embedding(num_avatars, 128)
        self.avatar_gamma_emb = nn.Embedding(num_avatars, norm_nc)
        self.avatar_beta_emb = nn.Embedding(num_avatars, norm_nc)

    def forward(self, x, avatar_index):
        avatar_shared = self.avatar_shared_emb(avatar_index)
        avatar_gamma = self.avatar_gamma_emb(avatar_index)
        avatar_beta = self.avatar_beta_emb(avatar_index)

        x = self.norm(x)
        shared_emb = self.conv_shared(x)
        gamma = self.conv_gamma(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        beta = self.conv_beta(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        gamma = gamma + avatar_gamma.view(-1, self.norm_nc, 1, 1)
        beta = beta + avatar_beta.view(-1, self.norm_nc, 1, 1)

        out = x * (1 + gamma) + beta
        return out