import torch
import torch.nn as nn
from torchinfo import summary

# Convolution for encoder 
# ResNet_18/34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        #--------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += shortcut
        out = self.relu(out)

        return out

# ResNet_50/101
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        #------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        #-------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                                kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out

# Deconvolution for decoder
class DeconvBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, expansion=1, stride=1, upsample=None):
        super(DeconvBasicBlock, self).__init__()
        self.expansion = expansion
        if stride == 1:
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                    kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            # Deconvolution top block
            self.conv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                                            kernel_size=3, stride=stride,
                                            padding=1, output_padding=1,
                                            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        #--------------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.upsample = upsample
    
    def forward(self, x):
        shortcut = x
        if self.upsample is not None:
            shortcut = self.upsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)

        return out


class DeconvBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        #------------------------------------------------------------
        if stride == 1:
            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                    kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            # Deconvolution top block
            self.conv2 = nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel,
                                            kernel_size=3, stride=stride, 
                                            padding=1, output_padding=1,
                                            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        #-----------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                                kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x
        if self.upsample is not None:
            shortcut = self.upsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)
    
        return out


class DTN_ResNet_AutoEncoder(nn.Module):
    # num_block[4] 存储4个layer的block数量
    def __init__(self, downblock, upblock, num_block, output_channels):
        super(DTN_ResNet_AutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                                kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # input channels for layer1
        self.in_channels = 64

        # 第二个参数决定该layer中第一个block的第一个卷积操作的输出channel数
        self.layer1 = self.make_downlayer(downblock, 64, num_block[0])
        self.layer2 = self.make_downlayer(downblock, 128, num_block[1], stride=2)
        self.layer3 = self.make_downlayer(downblock, 256, num_block[2], stride=2)
        self.layer4 = self.make_downlayer(downblock, 512, num_block[3], stride=2)  # output channels:2048

        self.uplayer1 = self.make_uplayer(upblock, 512, num_block[3], stride=2)  # 1024 14
        self.uplayer2 = self.make_uplayer(upblock, 256, num_block[2], stride=2)  # 512 28
        self.uplayer3 = self.make_uplayer(upblock, 128, num_block[1], stride=2)  # 256 56
        self.uplayer4 = self.make_uplayer(upblock, 64,  num_block[0], stride=2)  # output channels:128 112

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256? 128
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        ) 
        self.uplayer_top = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample)  # 64 224
        
        self.conv1_1 = nn.ConvTranspose2d(64, 3, kernel_size=1, stride=1, bias=False) # 3 224
        self.conv1_2 = nn.Conv2d(3, output_channels, kernel_size=3, stride=1, padding=1, bias=False) # 12 224
    

    def make_downlayer(self, block, init_channels, num_block, stride=1):
        # top block 
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=init_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        
        self.in_channels = init_channels * block.expansion
        
        for i in range(1, num_block):
            layers.append(block(self.in_channels, init_channels))
        
        return nn.Sequential(*layers)

    def make_uplayer(self, block, init_channels, num_block, stride=1):
        upsample = None
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels * 2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_block):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        
        return nn.Sequential(*layers)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def decoder(self, x): #output_size):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)

        x = self.conv1_1(x) #, output_size=output_size)
        x = self.conv1_2(x)
        return x

    
    def Expected_Texture_Transformer(self, features, texture):
        # slice on channels   features.shape(B,12,224,224)
        feature1 = features[:, 0 : 3, :, :]
        feature2 = features[:, 3 : 6, :, :]
        feature3 = features[:, 6 : 9, :, :]
        feature4 = features[:, 9 : 12, :, :]

        # Expected Texture Transformer
        transformer1 = self.relu(torch.sub(texture, feature1))
        transformer2 = torch.add(transformer1, feature2)
        transformer3 = torch.mul(transformer2, feature3)
        # transformer4 = torch.add(transformer3, feature4)
        transformer4 = self.relu(torch.add(transformer3, feature4))

        out = transformer4
        return out

    def forward(self, ref_image, texture):
        encoder_output = self.encoder(ref_image)
        decoder_output = self.decoder(encoder_output)  #, output_size=(1,12,224,224))
        rendered = self.Expected_Texture_Transformer(decoder_output, texture)
        return decoder_output, rendered


def DTN_ResNet50_AE(**kwargs):
    return DTN_ResNet_AutoEncoder(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 12, **kwargs)


def DTN_ResNet101_AE(**kwargs):
    return DTN_ResNet_AutoEncoder(Bottleneck, DeconvBottleneck, [3, 4, 23, 2], 12, **kwargs)


def DTN(ae_type):
    if ae_type == 'ResNet50':
        return DTN_ResNet_AutoEncoder(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 12)
    elif ae_type == 'ResNet101':
        return DTN_ResNet_AutoEncoder(Bottleneck, DeconvBottleneck, [3, 4, 23, 2], 12)
    else:
        print('no autoencoder model!')


if __name__ == "__main__":
    device = torch.device("cuda:0")
    # model = DTN(ae_type='ResNet50').to(device)
    model = DTN(ae_type='ResNet50').to(device=device)
    summary(model, [[1,3,448,448], [1,3,448,448]])







            
            





        