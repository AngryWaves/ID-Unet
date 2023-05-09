import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
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


class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


def make_layer(input_channels, output_channels, num_blocks=3, block=Res_CBAM_block):
    layers = []
    layers.append(block(input_channels, output_channels))
    for i in range(num_blocks - 1):
        layers.append(block(output_channels, output_channels))
    return nn.Sequential(*layers)


# 2023.3.24
class ID_UNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, m=3):
        super(ID_UNet, self).__init__()
        block = Res_CBAM_block
        # num_blocks = [3, 3, 3, 3]
        nb_filter = [16*x for x in [1, 2, 3, 4, 5]]
        # self.m = m
        #   encode
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)
        self.pool_8 = nn.MaxPool2d(8, 8)
        self.pool_16 = nn.MaxPool2d(16, 16)
        #   decode
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        #   encode 1
        self.conv1_0 = make_layer(input_channels, nb_filter[0], num_blocks=1)
        #   encode 2
        self.en2_conv1_0 = block(nb_filter[0], nb_filter[0])
        self.conv2_0 = make_layer(nb_filter[0], nb_filter[0], num_blocks=m)
        #   encode 3
        self.en3_conv1_0 = block(nb_filter[0], nb_filter[0])
        self.en3_conv2_0 = block(nb_filter[0], nb_filter[0])
        self.conv3_0 = make_layer(nb_filter[1], nb_filter[1], num_blocks=m)
        #   encode 4
        self.en4_conv1_0 = block(nb_filter[0], nb_filter[0])
        self.en4_conv2_0 = block(nb_filter[0], nb_filter[0])
        self.en4_conv3_0 = block(nb_filter[1], nb_filter[0])
        self.conv4_0 = make_layer(nb_filter[2], nb_filter[2], num_blocks=m)
        #   encode 5
        self.en5_conv1_0 = block(nb_filter[0], nb_filter[0])
        self.en5_conv2_0 = block(nb_filter[0], nb_filter[0])
        self.en5_conv3_0 = block(nb_filter[1], nb_filter[0])
        self.en5_conv4_0 = block(nb_filter[2], nb_filter[0])
        self.conv5_0 = make_layer(nb_filter[3], nb_filter[3], num_blocks=m)
        #   decode 4
        self.mid4_conv4_1 = block(nb_filter[2], nb_filter[0])
        self.de4_conv5_0 = block(nb_filter[3], nb_filter[0])
        self.conv4_1 = make_layer(nb_filter[1], nb_filter[1], num_blocks=m)
        #   decode 3
        self.mid3_conv3_2 = block(nb_filter[1], nb_filter[0])
        self.de3_conv4_1 = block(nb_filter[1], nb_filter[0])
        self.de3_conv5_0 = block(nb_filter[3], nb_filter[0])
        self.conv3_2 = make_layer(nb_filter[2], nb_filter[2], num_blocks=m)
        #   decode 2
        self.mid2_conv2_3 = block(nb_filter[0], nb_filter[0])
        self.de2_conv3_2 = block(nb_filter[2], nb_filter[0])
        self.de2_conv4_1 = block(nb_filter[1], nb_filter[0])
        self.de2_conv5_0 = block(nb_filter[3], nb_filter[0])
        self.conv2_3 = make_layer(nb_filter[3], nb_filter[3], num_blocks=m)
        #   decode 1
        self.mid1_conv1_4 = block(nb_filter[0], nb_filter[0])
        self.de1_conv2_3 = block(nb_filter[3], nb_filter[0])
        self.de1_conv3_2 = block(nb_filter[2], nb_filter[0])
        self.de1_conv4_1 = block(nb_filter[1], nb_filter[0])
        self.de1_conv5_0 = block(nb_filter[3], nb_filter[0])
        self.conv1_4 = make_layer(nb_filter[4], nb_filter[4], num_blocks=1)
        #   final
        self.final = nn.Conv2d(nb_filter[4], num_classes, kernel_size=1)

    def forward(self, input):
        # encode
        x1_0 = self.conv1_0(input)
        x2_0 = self.conv2_0(
            self.en2_conv1_0(self.pool_2(x1_0)))
        x3_0 = self.conv3_0(torch.cat([
            self.en3_conv1_0(self.pool_4(x1_0)),
            self.en3_conv2_0(self.pool_2(x2_0))], 1))
        x4_0 = self.conv4_0(torch.cat([
            self.en4_conv1_0(self.pool_8(x1_0)),
            self.en4_conv2_0(self.pool_4(x2_0)),
            self.en4_conv3_0(self.pool_2(x3_0))], 1))
        x5_0 = self.conv5_0(torch.cat([
            self.en5_conv1_0(self.pool_16(x1_0)),
            self.en5_conv2_0(self.pool_8(x2_0)),
            self.en5_conv3_0(self.pool_4(x3_0)),
            self.en5_conv4_0(self.pool_2(x4_0))], 1))
        # decode
        x4_1 = self.conv4_1(torch.cat([
            self.mid4_conv4_1(x4_0),
            self.de4_conv5_0(self.up_2(x5_0))], 1))
        x3_2 = self.conv3_2(torch.cat([
            self.mid3_conv3_2(x3_0),
            self.de3_conv4_1(self.up_2(x4_1)),
            self.de3_conv5_0(self.up_4(x5_0))], 1))
        x2_3 = self.conv2_3(torch.cat([
            self.mid2_conv2_3(x2_0),
            self.de2_conv3_2(self.up_2(x3_2)),
            self.de2_conv4_1(self.up_4(x4_1)),
            self.de2_conv5_0(self.up_8(x5_0))], 1))
        x1_4 = self.conv1_4(torch.cat([
            self.mid1_conv1_4(x1_0),
            self.de1_conv2_3(self.up_2(x2_3)),
            self.de1_conv3_2(self.up_4(x3_2)),
            self.de1_conv4_1(self.up_8(x4_1)),
            self.de1_conv5_0(self.up_16(x5_0))], 1))
        # final
        output = self.final(x1_4)
        return output


def main():
    model = ID_UNet()
    x = torch.ones((1, 3, 256, 256))
    print(model(x)[0].shape)


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
