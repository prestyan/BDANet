import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class DissimilarityLoss(nn.Module):
    def __init__(self):
        super(DissimilarityLoss, self).__init__()
        self.save_feature_num = 128
        self.fc = nn.Linear(200, self.save_feature_num)

    def forward(self, global_feature, local_feature):
        data_reduced = self.fc_dimension_reduction(local_feature)
        euclidean_distance = F.cosine_similarity(global_feature, data_reduced)
        # Maximizing the distance means minimizing the negative distance
        loss = -torch.mean(euclidean_distance)
        return loss

    def svd_dimension_reduction(self, feature):
        local_feature = feature.unsqueeze(1)
        U, S, V = torch.svd(local_feature)
        data_reduced = V[:, :self.save_feature_num, :].squeeze(2)
        return data_reduced

    def pca_dimension_reduction(self, feature):  # change 2, not recommend
        data_np = feature.cpu().detach().numpy()

        pca = PCA(n_components=self.save_feature_num)
        data_reduced_np = pca.fit_transform(data_np)

        data_reduced = torch.tensor(data_reduced_np, dtype=torch.float32)
        return data_reduced.cuda()

    def fc_dimension_reduction(self, feature):  # best but not recommend
        return self.fc(feature)

    def pool_dimension_reduction(self, x):
        # 计算池化窗口的大小
        kernel_size = 200 // self.save_feature_num
        stride = kernel_size
        # 使用平均池化
        pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        # 调整输入数据的形状为 [batch_size, channels, length]
        x = x.unsqueeze(1)
        # 应用池化
        x = pool(x)
        # 调整输出数据的形状为 [batch_size, output_size]
        x = x.squeeze(1)
        return x
