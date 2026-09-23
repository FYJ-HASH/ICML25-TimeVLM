import json
import matplotlib.pyplot as plt
import numpy as np

# ===== 修改这里：你的训练 epoch 数 =====
train_epochs = 20

# ===== 修改这里：SAM 历史数据文件路径 =====
# 该文件由训练流程在结束时自动生成（见 exp/exp_few_shot_forecasting.py），
# 内容结构为：
# {
#     "aps": [step_aps, ...],                  # 每个优化 step 的 APS 扰动幅度
#     "mdps": {
#         "modal1_multi_norm": [...],          # modal1(vision) 融合梯度范数
#         "modal1_uni_norm":   [...],          # modal1(vision) 单模态梯度范数
#         "modal2_multi_norm": [...],          # modal2(text)    融合梯度范数
#         "modal2_uni_norm":   [...]            # modal2(text)    单模态梯度范数
#     }
# }
SAM_HISTORY_PATH = 'sam_history_fusion_sam.json'


def load_sam_history(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_epoch_avg(data_list, train_epochs):
    """把按 step 记录的列表按 epoch 分段求平均，返回长度为 train_epochs 的列表。"""
    arr = np.asarray(data_list, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return [0.0] * train_epochs
    # 每个 epoch 平均分到的 step 数（最后一段可能略短）
    epoch_steps = n / train_epochs
    result = []
    for i in range(train_epochs):
        start = int(round(i * epoch_steps))
        end = int(round((i + 1) * epoch_steps))
        end = min(end, n)
        result.append(float(np.mean(arr[start:end])) if end > start else float(arr[-1]))
    return result


def plot_aps_curve(epochs, aps_epoch, save_path):
    """图1：APS 扰动幅度随每一轮 epoch 的变化。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, aps_epoch, color='#d62728', marker='o',
            linewidth=2, markersize=6, label='APS perturbation norm')
    ax.set_title('(a) APS Perturbation Magnitude', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perturbation Norm', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图1] APS 曲线已保存为 {save_path}")


def plot_mdps_curve(epochs, modal1_multi_epoch, modal1_uni_epoch,
                    modal2_multi_epoch, modal2_uni_epoch, save_path):
    """图2：MDPS 梯度分解随每一轮 epoch 的变化。"""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(epochs, modal1_multi_epoch, color='#1f77b4', marker='o',
            linewidth=2, markersize=5, label='modal1 (vision) in fusion')
    ax.plot(epochs, modal1_uni_epoch, color='#1f77b4', marker='s',
            linewidth=2, markersize=5, label='modal1 (vision) single')
    ax.plot(epochs, modal2_multi_epoch, color='#2ca02c', marker='o',
            linewidth=2, markersize=5, label='modal2 (text) in fusion')
    ax.plot(epochs, modal2_uni_epoch, color='#2ca02c', marker='s',
            linewidth=2, markersize=5, label='modal2 (text) single')
    ax.set_title('(b) MDPS Gradient Decomposition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图2] MDPS 曲线已保存为 {save_path}")


if __name__ == '__main__':
    # 加载 SAM 历史
    sam_data = load_sam_history(SAM_HISTORY_PATH)

    # 按 epoch 求平均
    aps_epoch = compute_epoch_avg(sam_data['aps'], train_epochs)
    modal1_multi_epoch = compute_epoch_avg(sam_data['mdps']['modal1_multi_norm'], train_epochs)
    modal1_uni_epoch = compute_epoch_avg(sam_data['mdps']['modal1_uni_norm'], train_epochs)
    modal2_multi_epoch = compute_epoch_avg(sam_data['mdps']['modal2_multi_norm'], train_epochs)
    modal2_uni_epoch = compute_epoch_avg(sam_data['mdps']['modal2_uni_norm'], train_epochs)

    epochs = list(range(1, train_epochs + 1))

    # 分别画两张独立的图
    plot_aps_curve(epochs, aps_epoch, 'aps_epoch_curve.png')
    plot_mdps_curve(epochs, modal1_multi_epoch, modal1_uni_epoch,
                    modal2_multi_epoch, modal2_uni_epoch, 'mdps_epoch_curve.png')

