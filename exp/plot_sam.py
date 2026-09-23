import json
import matplotlib.pyplot as plt
import numpy as np

# ===== 修改这里：你的训练 epoch 数 =====
train_epochs = 20

def load_sam_history(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_epoch_avg(data_list, train_epochs):
    epoch_steps = len(data_list) // train_epochs
    return [np.mean(data_list[i*epoch_steps:(i+1)*epoch_steps]) for i in range(train_epochs)]

# ===== 加载 SAM 历史 =====
sam_data = load_sam_history('sam_history_fusion_sam.json')

# 按 epoch 求平均
aps_epoch = compute_epoch_avg(sam_data['aps'], train_epochs)
modal1_multi_epoch = compute_epoch_avg(sam_data['mdps']['modal1_multi_norm'], train_epochs)
modal1_uni_epoch = compute_epoch_avg(sam_data['mdps']['modal1_uni_norm'], train_epochs)
modal2_multi_epoch = compute_epoch_avg(sam_data['mdps']['modal2_multi_norm'], train_epochs)
modal2_uni_epoch = compute_epoch_avg(sam_data['mdps']['modal2_uni_norm'], train_epochs)

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

epochs = range(1, train_epochs + 1)

# (a) APS 扰动幅度
axes[0].plot(epochs, aps_epoch, color='#d62728', marker='o', linewidth=2, markersize=5, label='APS perturbation norm')
axes[0].set_title('(a) APS Perturbation Magnitude', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Perturbation Norm', fontsize=12)
axes[0].legend(fontsize=10, loc='best')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xticks(epochs)

# (b) MDPS 梯度分解
axes[1].plot(epochs, modal1_multi_epoch, color='#1f77b4', marker='o', linewidth=2, markersize=5, label='modal1 (vision) in fusion')
axes[1].plot(epochs, modal1_uni_epoch, color='#1f77b4', marker='s', linewidth=2, markersize=5, label='modal1 (vision) single')
axes[1].plot(epochs, modal2_multi_epoch, color='#2ca02c', marker='o', linewidth=2, markersize=5, label='modal2 (text) in fusion')
axes[1].plot(epochs, modal2_uni_epoch, color='#2ca02c', marker='s', linewidth=2, markersize=5, label='modal2 (text) single')
axes[1].set_title('(b) MDPS Gradient Decomposition', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Gradient Norm', fontsize=12)
axes[1].legend(fontsize=9, loc='best')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xticks(epochs)

plt.tight_layout(pad=2.0)
plt.savefig('sam_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 sam_comparison.png")
plt.show()

