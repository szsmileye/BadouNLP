import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.rand(5)
    max_index = np.argmax(x)
    return x, max_index


def build_dataset(total_sample_number):
    X = []
    Y = []
    for _ in range(total_sample_number):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # X_np = np.array(X)
    # Y_np = np.array(Y).flatten()  # Flatten Y to ensure it's 1D
    # return torch.FloatTensor(X_np), torch.LongTensor(Y_np)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_number = 10
    x, y = build_dataset(test_sample_number)
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)  # Get the index of the max log-probability
        correct = (predicted == y).sum().item()
        print(y, predicted)
    accuracy = correct / test_sample_number
    print(f"Correct predictions: {correct}, Accuracy: {accuracy:.6f}")
    return accuracy
def main():
    # 配置参数
    epoch_number = 50 # 训练轮数
    batch_size = 10 # 每次训练的样本个数
    train_sample_number = 5000 # 每轮训练的样本总数
    input_size = 5 # 输入向量纬度
    learning_rate = 1e-3 # 学习率 10的-3次方 0.001
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log =[]
    # 创建训练集
    train_x,train_y = build_dataset(train_sample_number)
    # 训练工程
    for epoch in range(epoch_number):
        model.train()
        # 检测损失
        watch_loss = []
        for batch_index in range(train_sample_number // batch_size): # train_sample_number // batch_size 计算了总共有多少个批次。这里使用的是整数除法（//），意味着结果会被向下取整到最接近的整数。如果 train_sample_number 不是 batch_size 的整数倍，那么最后一个批次将包含少于 batch_size 的样本。
            # 分页获取集合中的值
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            # 计算loss
            loss = model.forward(x,y) # 等价于 model(x,y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optimizer.step()
            # 梯度归零
            optimizer.zero_grad()
            # 记录loss
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 测试本轮模型结果
        acc = evaluate(model)
        # 记录结果
        log.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    # 画acc曲线
    plt.plot(range(len(log)), [l[0] for l in log],label="acc")
    # 画loss曲线
    plt.plot(range(len(log)),[l[1] for l in log ],label="loss")

    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()
    return

if __name__ == "__main__":
    main()















