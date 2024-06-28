import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载特征数组
features = np.load('./results/ours/feature/f3.npy')

# 检查数据的形状
print("特征数组的形状:", features.shape)  # 应该是 (447, 328)

# 使用t-SNE进行降维到2维
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 将数据分成三份
num_samples = features.shape[0]
indices = np.arange(num_samples)
# np.random.shuffle(indices)

split1 = indices[:num_samples // 3]
split2 = indices[num_samples // 3: 2 * num_samples // 3]
split3 = indices[2 * num_samples // 3:]

# 可视化
plt.figure(figsize=(10, 6))

plt.scatter(features_2d[split1, 0], features_2d[split1, 1], s=40, color='#1f77b4', label='Underload', alpha=1.)  # 柔和的蓝色
plt.scatter(features_2d[split3, 0], features_2d[split3, 1], s=40, color='#ff7f0e', label='Overload', alpha=1.)  # 橙色
plt.scatter(features_2d[split2, 0], features_2d[split2, 1], s=40, color='#2ca02c', label='Normal', alpha=1.)  # 深绿色



plt.title('Visualization of BDANet Features', fontsize=18)
#plt.xlabel('t-SNE feature 1', fontsize=14)
#plt.ylabel('t-SNE feature 2', fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.show()
