import json
import matplotlib.pyplot as plt
import numpy as np

# 加载 loss 历史
with open('loss_history_xxx.json', 'r') as f:
    loss_history = json.load(f)

# 每个 epoch 的 loss（取平均值）
temporal_loss = loss_history['temporal']
multimodal_loss = loss_history['multimodal']
fusion_loss = loss_history['fusion']

# 按 epoch 求平均
epoch_steps = len(temporal_loss) // train_epochs
temporal_epoch = [np.mean(temporal_loss[i*epoch_steps:(i+1)*epoch_steps]) for i in range(train_epochs)]
multimodal_epoch = [np.mean(multimodal_loss[i*epoch_steps:(i+1)*epoch_steps]) for i in range(train_epochs)]
fusion_epoch = [np.mean(fusion_loss[i*epoch_steps:(i+1)*epoch_steps]) for i in range(train_epochs)]

# 画图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 时序分支
axes[0].plot(temporal_epoch, 'r-o', label='temporal-only model')
axes[0].set_title('(a) Temporal Modality')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) 多模态分支
axes[1].plot(multimodal_epoch, 'g-o', label='multimodal in fusion model')
axes[1].set_title('(b) Multimodal Modality')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# (c) 融合后
axes[2].plot(fusion_epoch, 'b-o', label='fusion model')
axes[2].set_title('(c) Fusion')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modality_comparison.png', dpi=300)
plt.show()
