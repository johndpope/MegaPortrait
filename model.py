import gc
import torch
import torch.nn as nn
import torch.nn.functional as F



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