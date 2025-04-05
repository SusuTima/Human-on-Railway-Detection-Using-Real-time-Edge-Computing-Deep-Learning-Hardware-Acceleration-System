# 初始化模型結構
model = LeNet()
model.load_state_dict(torch.load("/content/lenet_human_detection.pth"))
model.eval()  # 切換到評估模式

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms  # 確保引入 transforms
def test_model(model, data_folder, classes):
    """
    測試模型對資料夾中所有圖片的分類準確率，並打印每張圖片的預測類別與機率。

    :param model: 已加載的 PyTorch 模型
    :param data_folder: 主資料夾路徑，包含子資料夾作為類別
    :param classes: 類別列表（子資料夾名稱）
    :return: None
    """
    # 定義預處理流程
    transform = transforms.Compose([
        transforms.Resize((80, 80)),  # 調整大小
        transforms.ToTensor(),       # 轉為 Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 標準化
    ])

    correct = 0
    total = 0

    # 設置模型為評估模式
    model.eval()

    with torch.no_grad():  # 禁用梯度計算以加速
        for class_index, class_name in enumerate(classes):
            class_folder = os.path.join(data_folder, class_name)
            if not os.path.exists(class_folder):
                print(f"警告: 資料夾 {class_folder} 不存在，跳過。")
                continue

            for filename in os.listdir(class_folder):
                file_path = os.path.join(class_folder, filename)
                try:
                    # 載入和預處理圖片
                    image = Image.open(file_path).convert("RGB")
                    input_tensor = transform(image).unsqueeze(0)  # 增加 batch 維度

                    # 模型推論
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)  # 計算機率
                    _, predicted = torch.max(probabilities, 1)

                    # 比較預測結果與真實類別
                    if predicted.item() == class_index:
                        correct += 1
                    total += 1

                    # 打印每張圖片的結果
                    predicted_class = classes[predicted.item()]
                    confidence = probabilities[0, predicted.item()].item()
                    print(f"圖片: {file_path}, 預測類別: {predicted_class}, 機率: {confidence:.4f}")

                except Exception as e:
                    print(f"無法處理圖片 {file_path}: {e}")

    # 計算準確率
    accuracy = correct / total if total > 0 else 0
    print(f"\n總準確率: {accuracy * 100:.2f}% ({correct}/{total})")

# 初始化模型並加載權重
model = LeNet()
weight_path = "/content/lenet_human_detection.pth"
model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

# 測試資料夾
data_folder = "/content/sample_data/test_data"  # 替換為你的資料夾路徑
classes = ["human", "no_human"]     # 子資料夾名稱對應類別

# 測試模型並輸出準確率
test_model(model, data_folder, classes)
