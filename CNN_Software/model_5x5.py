import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一層卷積：加上 padding=2
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # 輸入3個通道，輸出6個通道
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二層卷積：不加 padding
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 輸入6個通道，輸出16個通道
        # 第三層卷積：不加 padding
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5)  # 輸入16個通道，輸出16個通道
        # 單層全連接層，輸入大小為16 * 7 * 7，輸出大小為2
        self.fc1 = nn.Linear(16 * 7 * 7, 2)  # 二元分類

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一層卷積+池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二層卷積+池化
        x = self.pool(F.relu(self.conv3(x)))  # 第三層卷積+池化
        x = torch.flatten(x, 1)  # 扁平化輸入
        x = self.fc1(x)  # 單層全連接層輸出
        return x

# 初始化模型
model = LeNet()
print(model)
