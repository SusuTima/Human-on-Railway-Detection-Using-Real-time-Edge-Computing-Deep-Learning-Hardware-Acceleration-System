#替換損失函數與優化器
import torch
import torch.nn as nn
import torch.optim as optim
# 定義損失函數與優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#訓練和測試程式碼
import matplotlib.pyplot as plt

# 初始化記錄清單
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 訓練模型
for epoch in range(20):  # 訓練 20 個世代
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 累積損失
        running_loss += loss.item()
        
        # 計算準確率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 記錄訓練損失和準確率
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # 測試模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 計算準確率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 記錄測試損失和準確率
    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(correct / total)
    
    # 打印每個 epoch 的結果
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

# 繪製圖表
plt.figure(figsize=(12, 6))

# Loss 曲線
plt.subplot(1, 2, 1)
plt.plot(range(1, 21), train_losses, label='Train Loss')
plt.plot(range(1, 21), test_losses, label='Test Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('model loss')
plt.legend()

# Accuracy 曲線
plt.subplot(1, 2, 2)
plt.plot(range(1, 21), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 21), test_accuracies, label='Test Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#驗證模型性能
from sklearn.metrics import accuracy_score, classification_report
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Classification Report:\n", classification_report(all_labels, all_preds))

#載入模型
torch.save(model.state_dict(), "lenet_human_detection.pth")
