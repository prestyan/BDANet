# 导入必要的库
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import EegDataset
from network import MBiLstmDcnn
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time


# 训练模型
def interaug(timg, label, batch_size, num):
    aug_data = []
    aug_label = []
    for cls4aug in range(3):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        tmp_aug_data = np.zeros((int(batch_size / 3), 1, 61, 500))
        for ri in range(int(batch_size / 3)):
            for rj in range(5):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                tmp_aug_data[ri, :, :, rj * 100:(rj + 1) * 100] = tmp_data[rand_idx[rj], :, :,
                                                                  rj * 100:(rj + 1) * 100]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / 3)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data).cuda()
    aug_data = aug_data.float()
    aug_label = torch.from_numpy(aug_label).cuda()
    aug_label = aug_label.long()
    # return aug_data, aug_label
    return aug_data[:num], aug_label[:num]


def train_and_val(train_loader, test_loader, nSub, batch_size, X_train, Y_train):
    log_write = open("./results/ours/log_subject%d.txt" % nSub, "w")
    # 创建模型实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MBiLstmDcnn(input_size, hidden_size, num_layers, num_classes, num_heads, attention_dim, feedforward_dim,
                        0.5, 0.5, dropout_rate).to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # 使用adam优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    bestAcc = 0
    bestf1 = 0
    Y_true = 0
    Y_pred = 0
    model_dict = 0
    print("Subjetc_" + str(nSub))
    for epoch in range(num_epochs):
        # 遍历数据集
        model.train()
        total_loss = 0
        st = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据和标签转移到设备上
            inputs = inputs.to(device)
            labels = labels.to(device)
            # S&R 数据增强
            # if len(inputs) == batch_size:
            #    print(len(inputs))
            aug_data, aug_label = interaug(X_train, Y_train, batch_size, 40)
            inputs = torch.cat((inputs, aug_data))
            labels = torch.cat((labels, aug_label))

            # 前向传播
            optimizer.zero_grad()
            outputs, reg_loss = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)
            # loss = loss + reg_loss
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            # scheduler.step()
            total_loss = total_loss + loss.item()
        # 测试模型
        et = time.time()
        print("Train per epoch(357 samples): %.2f s" % (et - st))
        model.eval()
        with torch.no_grad():
            # 初始化正确和总数
            correct = 0
            total = 0
            first = True
            # 遍历测试集
            start_time = time.time()
            for inputs, labels in test_loader:
                # 将数据和标签转移到设备上
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs, _ = model(inputs)

                # 预测类别
                _, predicted = torch.max(outputs, 1)

                # 更新正确和总数
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if first:
                    predy = predicted
                    truey = labels
                    first = False
                else:
                    predy = torch.cat((predy, predicted))
                    truey = torch.cat((truey, labels))
            # 计算并打印准确率
            accuracy = 100 * correct / total
            f1 = 100 * f1_score(truey.cpu(), predy.cpu(), average='macro')
        if accuracy > bestAcc:
            bestAcc = accuracy
            bestf1 = f1
            Y_true = truey
            Y_pred = predy
            model_dict = model.state_dict()
        end_time = time.time()
        # print('A test/single subject time : ', str(end_time - start_time), "s / ", str((end_time - start_time)/90))
        print(f'Epoch {epoch + 1}, Train Loss {total_loss / len(train_loader):.6f}, ',
              f'Accuracy/f1 of the model on the test set: {accuracy:.2f}% / {f1:.2f}%')
        log_write.write(str(epoch) + "   " + str(accuracy) + "   " + str(f1) + "\n")
    log_write.write('The best accuracy/f1-score is: ' + str(bestAcc) + "  " + str(bestf1) + "\n")
    print('The best accuracy/f1-score is: ' + str(bestAcc) + "  " + str(bestf1))
    torch.save(model_dict, './results/ours/model/lstmdcnn_' + str(nSub) + '.pth')
    path = 'D:/研究生/研一下/lstmcnn/code/results/ours/cm/Subject' + str(nSub) + '.png'
    conf_matrix(Y_pred.cpu(), Y_true.cpu(), path, nSub)
    return bestAcc, bestf1


def conf_matrix(pred_labels, true_labels, path, nsub):
    cm = confusion_matrix(true_labels, pred_labels)
    class_labels = ['Left_Hands', 'Right_Hands', 'Feet', 'Tongue']
    class_labels_r2 = ['Underload', 'Normal', 'Overload']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Blues
    plt.title('Subject ' + str(nsub))
    plt.colorbar()
    tick_marks = np.arange(len(class_labels_r2))
    plt.xticks(tick_marks, class_labels_r2, rotation=45)
    plt.yticks(tick_marks, class_labels_r2)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="black" if cm[i, j] < thresh else "white",
                 fontsize=20)

    plt.xlabel('Pred labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def get_channel_attention(feature_map):
    model = MBiLstmDcnn(input_size, hidden_size, num_layers, num_classes, num_heads, attention_dim, feedforward_dim,
                        0.5, 0.5, dropout_rate).to(torch.device("cuda"))
    model.load_state_dict(torch.load('./results/ours/S1/model/lstmdcnn_1.pth'))
    _, channel_attention = model(feature_map)
    channel_attention_numpy = channel_attention.cpu().detach().numpy()

    # 保存为.mat格式
    print(channel_attention_numpy.shape)
    scipy.io.savemat('channel_attention.mat', {'channel_attention': channel_attention_numpy})


if __name__ == "__main__":

    # 定义超参数
    input_size = 61  # 输入的维度
    hidden_size = 64  # 隐藏层的维度
    num_layers = 1  # lstm的层数
    num_classes = 3  # 分类的类别数
    batch_size = 16  # 批次大小
    learning_rate = 0.0003  # 学习率
    num_epochs = 200  # 训练的轮数
    num_heads = 4  # 头的数量
    attention_dim = 64  # 注意力维度
    feedforward_dim = 128  # 前馈网络维度
    dropout_rate = 0.3  # 丢弃率
    log_path = 'D:/研究生/研一下/lstmcnn/code/subject_log.txt'
    root = 'E:/data/EEG/Sub_S1_single/Sub_S1_'
    log = open(log_path, 'w')
    acc_res, f1_res = [], []
    for i in [1]:
        # 先生成一个随机数作为种子, 然后再用这个种子生成多个随机数
        # seed_n = np.random.randint(2023)
        seed_n = 1080
        print('seed is ' + str(seed_n))
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)  # cpu
        torch.cuda.manual_seed(seed_n)  # gpu
        torch.cuda.manual_seed_all(seed_n)  # 所有gpu

        data = scipy.io.loadmat(root + str(i) + '.mat')
        S1Data = data['S1Data']  # shape: (61, 500, 447)
        S1Data = np.transpose(S1Data, (2, 0, 1))
        S1Data = np.expand_dims(S1Data, axis=1)  # shape:(447, 1, 61, 500)
        # 获得attention
        # get_channel_attention(torch.from_numpy(S1Data).float().cuda())
        # sys.exit()
        # 获得完成
        S1Label = data['S1Label']  # shape: (447, 1)
        X_train, X_test, y_train, y_test = train_test_split(S1Data, S1Label, test_size=0.2, random_state=42)
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        train_dataset = EegDataset(X_train, y_train, 's')
        test_dataset = EegDataset(X_test, y_test, 's')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        acc, f1 = train_and_val(train_loader, test_loader, i, batch_size, X_train, y_train)
        acc_res.append(acc)
        f1_res.append(f1)
        # print("best_value, acc: ", max(acc_res), " ,f1: ", max(f1_res))
        log.write("Subject " + str(i) + "   " + str(acc) + "   " + str(f1) + "\n")
    log.write("average_value, acc:" + str(sum(acc_res) / len(acc_res)) + " ,f1: " + str(sum(f1_res) / len(f1_res)))
    print("average_value, acc:", sum(acc_res) / len(acc_res), " ,f1: ", sum(f1_res) / len(f1_res))
