import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetModified(nn.Module):
    def __init__(self):
        super(LeNetModified, self).__init__()
        # 第一層卷積：5x5 換成兩層 3x3
        self.conv1_1 = nn.Conv2d(3, 6, kernel_size=3, padding=2)  # 第一層 3x3
        self.conv1_2 = nn.Conv2d(6, 6, kernel_size=3)  # 第二層 3x3

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二層卷積：5x5 換成兩層 3x3
        self.conv2_1 = nn.Conv2d(6, 16, kernel_size=3)  # 第一層 3x3
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3)  # 第二層 3x3

        # 第三層卷積：5x5 換成兩層 3x3
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size=3)  # 第一層 3x3
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size=3)  # 第二層 3x3

        # 單層全連接層，輸入大小為16 * 7 * 7，輸出大小為2
        self.fc1 = nn.Linear(16 * 7 * 7, 2)  # 二元分類

    def forward(self, x):
        # 第一層卷積（兩層 3x3）+ 池化
        x = F.relu(self.conv1_1(x))
        x = self.pool(F.relu(self.conv1_2(x)))

        # 第二層卷積（兩層 3x3）+ 池化
        x = F.relu(self.conv2_1(x))
        x = self.pool(F.relu(self.conv2_2(x)))

        # 第三層卷積（兩層 3x3）+ 池化
        x = F.relu(self.conv3_1(x))
        x = self.pool(F.relu(self.conv3_2(x)))

        # 扁平化輸入
        x = torch.flatten(x, 1)

        # 單層全連接層輸出
        x = self.fc1(x)
        return x

# 初始化模型
model = LeNetModified()
