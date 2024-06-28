import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from MHA import ChannelWiseAttention, SELayer, MDAOutput, TransformerEncoder
from einops import rearrange


class ResidualAdd(nn.Module):
    def __init__(self, hidden_size, num_heads, att_drop, bf):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_size, 8, att_drop, batch_first=True)

    def forward(self, x, **kwargs):
        res, w = self.multihead_attn(x, x, x)
        x += res
        return x, w


class MBiLstmDcnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 num_heads, attention_dim, feedforward_dim, lstm_drop=0.25, dcnn_drop=0.25, att_drop=0.1):
        super(MBiLstmDcnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.feedforward_dim = feedforward_dim
        self.lstm_drop_p = lstm_drop
        self.dc_drop_p = dcnn_drop
        self.att_drop = att_drop

        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True, bidirectional=True)  # 使用lstm层
        self.lstm2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=int(hidden_size / 2),
                             num_layers=num_layers,
                             batch_first=True, bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, 8, att_drop, batch_first=True)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        # self.lstm_attn = MDAOutput(int(hidden_size / 2), int(hidden_size / 2), int(hidden_size / 2))

        # Conv Pool Block-1
        self.conv11 = nn.Conv2d(1, 25, (1, 10), padding=0)
        self.conv12 = nn.Conv2d(25, 25, (61, 1), padding=0)  # [bs,25,61,491]->[bs,25,1,491] and next->[bs,1,25,491]
        self.bn1 = nn.BatchNorm2d(25, False)
        self.pooling1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.se1 = SELayer(25)

        # Conv Pool Block-2
        self.conv2 = nn.Conv2d(25, 50, (1, 10), padding=0)  # [bs,1,25,163]->[bs,50,1,154] and next->[bs,1,50,154]
        self.bn2 = nn.BatchNorm2d(50)
        self.pooling2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.se2 = SELayer(50)

        # Conv Pool Block-3
        self.conv3 = nn.Conv2d(50, 100, (1, 10), padding=0)  # [bs,1,50,51]->[bs,100,1,42] and next->[bs,1,100,42]
        self.bn3 = nn.BatchNorm2d(100)
        self.pooling3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.se3 = SELayer(100)

        # Conv Pool Block-4
        self.conv4 = nn.Conv2d(100, 200, (1, 10), padding=0)  # [bs,1,100,14]->[bs,200,1,5] and next->[bs,1,200,5]
        self.bn4 = nn.BatchNorm2d(200)
        self.pooling4 = nn.MaxPool2d((1, 3), stride=(1, 3))

        self.fc1 = nn.Linear(200 + int(self.hidden_size / 2 * 4), 128)  # 使用全连接层进行分类 200 + int(self.hidden_size / 2 * 4)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()  # Relu
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.dc_drop_p)
        self.transEncoder = TransformerEncoder(depth=3, emb_size=32)

        self.fcd = nn.Dropout(0.5)
        self.fcd2 = nn.Dropout(0.3)

        # self.multi_head_attention_layer = MultiHeadAttentionLayer(self.num_heads, self.attention_dim, self.att_drop)
        # self.feed_forward_layer = FeedForwardLayer(self.attention_dim, self.feedforward_dim, self.att_drop)
        self.channel_attention = ChannelWiseAttention(61)

    def forward(self, x):
        x2 = x
        x = x.permute(0, 2, 1, 3)
        x, reg_loss = self.channel_attention(x)
        x = (x.squeeze(2)).permute(0, 2, 1)
        x_slice = torch.split(x, 125, dim=1)  # 按照第二个维度切分张量
        x1_out = torch.empty(x.size(0), 0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # ht_out = torch.empty(x.size(0), 0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for x1 in x_slice:
            x1, (ht1, ct1) = self.lstm1(x1)
            x1 = self.relu(x1)  # remove drop
            x1 = x1.contiguous().view(-1, 125, 2, self.hidden_size)
            x1 = torch.mean(x1, dim=2)
            # x1 = self.LayerNorm(x1)
            x1, w = self.multihead_attn(x1, x1, x1)
            x1, (ht2, ct2) = self.lstm2(x1)
            x1 = self.lstmDrop(self.relu(x1))
            # 同时利用前向和后向的输出，将它们从中间切割，然后求平均
            x1 = x1.contiguous().view(-1, 125, 2, int(self.hidden_size / 2))
            x1 = torch.mean(x1, dim=2)  # [bs, 125, 32]
            # x1 = self.lstm_attn(x1)
            x1 = x1[:, -1, :]
            x1 = x1.view(x1.size(0), -1)
            x1_out = torch.cat((x1_out, x1), dim=1)
        # x1_out = self.transEncoder(x1_out)
        # x1_out = x1_out[:, [124, 249, 374, 499], :]
        # x1_out = rearrange(x1_out, 'b t f -> b (t f)')

        # Layer 1
        x2 = self.pooling1(self.relu(self.bn1(self.conv12(self.conv11(x2)))))
        # x2 = self.convDrop(x2)
        x2 = self.se1(x2)

        # Layer 2
        x2 = self.pooling2(self.relu(self.bn2(self.conv2(x2))))
        # x2 = self.convDrop(x2)
        x2 = self.se2(x2)

        # Layer 3
        x2 = self.pooling3(self.relu(self.bn3(self.conv3(x2))))
        # x2 = self.convDrop(x2)
        x2 = self.se3(x2)

        # Layer 4, here can change to transformer
        x2 = self.pooling4(self.relu(self.bn4(self.conv4(x2))))
        x2 = self.convDrop(x2)
        x2 = x2.view(x2.size(0), -1)

        x_all = torch.cat((x1_out, x2), dim=1)
        self.x1_out = x1_out
        self.x2 = x2
        self.features = x_all

        x_out = self.fcd(self.relu(self.fc1(x_all)))
        self.features2 = x_out
        x_out = self.fcd2(self.relu(self.fc2(x_out)))
        x_out = self.fc3(x_out)
        return F.softmax(x_out, dim=1), reg_loss

    def getF(self, index=0):
        if index == 0:
            return self.features
        print('f->2')
        return self.features2

    def get_x1_and_x2(self):
        return self.x1_out, self.x2


class MBiLstmDcnn2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 num_heads, attention_dim, feedforward_dim, lstm_drop=0.25, dcnn_drop=0.25, att_drop=0.1):
        super(MBiLstmDcnn2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.feedforward_dim = feedforward_dim
        self.lstm_drop_p = lstm_drop
        self.dc_drop_p = dcnn_drop
        self.att_drop = att_drop

        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True, bidirectional=True)  # 使用lstm层
        self.lstm2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=int(hidden_size / 2),
                             num_layers=num_layers,
                             batch_first=True, bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, 8, 0.1, batch_first=True)
        # self.lstm_attn = MDAOutput(int(hidden_size / 2), int(hidden_size / 2), int(hidden_size / 2))

        # Conv Pool Block-1
        self.conv11 = nn.Conv2d(1, 25, (1, 10), padding=0)
        self.conv12 = nn.Conv2d(25, 25, (61, 1), padding=0)  # [bs,25,61,491]->[bs,25,1,491] and next->[bs,1,25,491]
        self.bn1 = nn.BatchNorm2d(25, False)
        self.pooling1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.se1 = SELayer(25)

        # Conv Pool Block-2
        self.conv2 = nn.Conv2d(25, 50, (1, 10), padding=0)  # [bs,1,25,163]->[bs,50,1,154] and next->[bs,1,50,154]
        self.bn2 = nn.BatchNorm2d(50)
        self.pooling2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.se2 = SELayer(50)

        # Conv Pool Block-3
        self.conv3 = nn.Conv2d(50, 100, (1, 10), padding=0)  # [bs,1,50,51]->[bs,100,1,42] and next->[bs,1,100,42]
        self.bn3 = nn.BatchNorm2d(100)
        self.pooling3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.se3 = SELayer(100)

        # Conv Pool Block-4
        self.conv4 = nn.Conv2d(100, 200, (1, 10), padding=0)  # [bs,1,100,14]->[bs,200,1,5] and next->[bs,1,200,5]
        self.bn4 = nn.BatchNorm2d(200)
        self.pooling4 = nn.MaxPool2d((1, 3), stride=(1, 3))

        self.fc1 = nn.Linear(200 + int(self.hidden_size / 2 * 4), 128)  # 使用全连接层进行分类
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()  # Relu
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.dc_drop_p)
        self.transEncoder = TransformerEncoder(depth=3, emb_size=32)

        self.fcd = nn.Dropout(0.5)
        self.fcd2 = nn.Dropout(0.3)

        # self.multi_head_attention_layer = MultiHeadAttentionLayer(self.num_heads, self.attention_dim, self.att_drop)
        # self.feed_forward_layer = FeedForwardLayer(self.attention_dim, self.feedforward_dim, self.att_drop)
        self.channel_attention = ChannelWiseAttention(61)

    def forward(self, x):
        # x = (x.unsqueeze(2)).permute(0, 3, 2, 1)
        x = x.permute(0, 2, 1, 3)
        x, reg_loss = self.channel_attention(x)
        x = (x.squeeze(2)).permute(0, 2, 1)
        x_slice = torch.split(x, 125, dim=1)  # 按照第二个维度切分张量
        x1_out = torch.empty(x.size(0), 0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # ht_out = torch.empty(x.size(0), 0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for x1 in x_slice:
            x1, (ht1, ct1) = self.lstm1(x1)
            x1 = self.relu(x1)  # remove drop
            x1 = x1.contiguous().view(-1, 125, 2, self.hidden_size)
            x1 = torch.mean(x1, dim=2)
            x1, w = self.multihead_attn(x1, x1, x1)
            x1, (ht2, ct2) = self.lstm2(x1)
            x1 = self.lstmDrop(self.relu(x1))
            # 同时利用前向和后向的输出，将它们从中间切割，然后求平均
            x1 = x1.contiguous().view(-1, 125, 2, int(self.hidden_size / 2))
            x1 = torch.mean(x1, dim=2)  # [bs, 125, 32]
            # x1 = self.lstm_attn(x1)
            x1 = x1[:, -1, :]
            x1 = x1.view(x1.size(0), -1)
            x1_out = torch.cat((x1_out, x1), dim=1)
        # x1_out = self.transEncoder(x1_out)
        # x1_out = x1_out[:, [124, 249, 374, 499], :]
        # x1_out = rearrange(x1_out, 'b t f -> b (t f)')

        x2 = (x.unsqueeze(-1)).permute(0, 3, 2, 1)
        # Layer 1
        x2 = self.pooling1(self.relu(self.bn1(self.conv12(self.conv11(x2)))))
        # x2 = self.convDrop(x2)
        x2 = self.se1(x2)

        # Layer 2
        x2 = self.pooling2(self.relu(self.bn2(self.conv2(x2))))
        # x2 = self.convDrop(x2)
        x2 = self.se2(x2)

        # Layer 3
        x2 = self.pooling3(self.relu(self.bn3(self.conv3(x2))))
        # x2 = self.convDrop(x2)
        x2 = self.se3(x2)

        # Layer 4, here can change to transformer
        x2 = self.pooling4(self.relu(self.bn4(self.conv4(x2))))
        x2 = self.convDrop(x2)
        x2 = x2.view(x2.size(0), -1)

        x_all = torch.cat((x1_out, x2), dim=1)
        self.features = x_all

        x_out = self.fcd(self.relu(self.fc1(x_all)))
        x_out = self.fcd2(self.relu(self.fc2(x_out)))
        x_out = self.fc3(x_out)
        return F.softmax(x_out, dim=1), reg_loss

    def getF(self):
        return self.features
