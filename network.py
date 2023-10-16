#####################################
## Unet模型 ##
######################################

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_channels, num_filters) -> None:
        super(Encoder, self).__init__()  # 继承父类的所有属性
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn2 = nn.BatchNorm2d(num_features=num_filters)
        self.relu2 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x_conv = x  # 保存下来，后面会拼接

        x_pool = self.maxpool(x)  # 这个是往下继续传递的

        return x_conv, x_pool


class Decoder(nn.Module):
    def __init__(self, num_channels, num_filters) -> None:
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        # 图片尺寸变大一倍

        self.conv1 = nn.Conv2d(
            in_channels=num_filters * 2,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn2 = nn.BatchNorm2d(num_features=num_filters)

        self.relu2 = nn.ReLU()

    def forward(self, input_conv, input_pool):
        x = self.up(input_pool)  # 上采样

        # 计算差距，featuremap 为 C*H*W
        h_diff = input_conv.size()[2] - x.size()[2]
        w_diff = input_conv.size()[3] - x.size()[3]

        pad = nn.ConstantPad2d(
            (h_diff // 2, h_diff - h_diff // 2, w_diff // 2, w_diff - w_diff // 2), 0
        )
        x = pad(x)

        # 将上采样后的图片与之前保存的图片进行拼接
        x = torch.cat((x, input_conv), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu2(x)


class Unet(nn.Module):
    def __init__(self, num_classes=59) -> None:
        super(Unet, self).__init__()

        self.down1 = Encoder(num_channels=3, num_filters=64)
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)

        self.mid_conv1 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
        )

        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.relu1 = nn.ReLU()
        self.mid_conv2 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1
        )

        self.bn2 = nn.BatchNorm2d(num_features=1024)
        self.relu2 = nn.ReLU()

        self.up1 = Decoder(num_channels=1024, num_filters=512)
        self.up2 = Decoder(num_channels=512, num_filters=256)
        self.up3 = Decoder(num_channels=256, num_filters=128)
        self.up4 = Decoder(num_channels=128, num_filters=64)

        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x_conv1, x_pool1 = self.down1(x)
        x_conv2, x_pool2 = self.down2(x_pool1)
        x_conv3, x_pool3 = self.down3(x_pool2)
        x_conv4, x_pool4 = self.down4(x_pool3)

        x_mid = self.mid_conv1(x_pool4)
        x_mid = self.bn1(x_mid)
        x_mid = self.relu1(x_mid)
        x_mid = self.mid_conv2(x_mid)
        x_mid = self.bn2(x_mid)
        x_mid = self.relu2(x_mid)

        x = self.up1(x_conv4, x_mid)
        x = self.up2(x_conv3, x)
        x = self.up3(x_conv2, x)
        x = self.up4(x_conv1, x)

        return self.out(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    model = Unet()
    y = model(x)
    print(y.shape)
