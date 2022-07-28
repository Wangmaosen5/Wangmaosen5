from torch import nn

out = [32, 64, 128, 64, 32]
class BasicBlock(nn.Module):
    def __init__(self, anchors_mask):
        super(BasicBlock, self).__init__()
        # out = [3, 32, 64, 128, 64, len(anchors_mask[0]) * 5]
        # out = [32, 64, 128, 64, len(anchors_mask[0]) * 5]
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=out[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(out[0])
        self.relu0 = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(out[0], out[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out[1])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(out[1], out[2], kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out[2])
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(out[2], out[3], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out[3])
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(out[3], out[4], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out[4])
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(out[4], len(anchors_mask[0]) * 5, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(len(anchors_mask[0]) * 5)
        self.relu5 = nn.LeakyReLU(0.1)

    def forward(self, x):
        # residual = x
        x = self.relu0(self.bn0(self.conv0(x)))
        for i in range(0, 4):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            # x += residual
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.relu4(self.bn4(self.conv4(x)))
            if i == 2:
                x52 = x
            if i == 3:
                x26 = x
            if i == 4:
                x13 = x
        x52 = self.relu5(self.bn5(self.conv5(x52)))
        x26 = self.relu5(self.bn5(self.conv5(x26)))
        x13 = self.relu5(self.bn5(self.conv5(x13)))
        return x13, x26, x52