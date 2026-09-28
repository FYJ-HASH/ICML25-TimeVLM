import json
import matplotlib.pyplot as plt
import numpy as np

# ===== 按你的训练结果填好 =====
train_epochs = 20
SAM_HISTORY_PATH = '../sam_history_fusion_sam.json'


def load_sam_history(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_epoch_avg(data_list, train_epochs):
    arr = np.asarray(data_list, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return [0.0] * train_epochs
    epoch_steps = n / train_epochs
    result = []
    for i in range(train_epochs):
        start = int(round(i * epoch_steps))
        end = int(round((i + 1) * epoch_steps))
        end = min(end, n)
        result.append(float(np.mean(arr[start:end])) if end > start else float(arr[-1]))
    return result


def plot_aps_score_curve(epochs, aps_score, save_path):
    """图1：APS 分数曲线（论文版）"""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(epochs, aps_score['vision'], color='#d62728', marker='o',
            linewidth=2, markersize=5, label='vision')
    ax.plot(epochs, aps_score['text'], color='#1f77b4', marker='s',
            linewidth=2, markersize=5, label='text')
    ax.plot(epochs, aps_score['temporal'], color='#2ca02c', marker='^',
            linewidth=2, markersize=5, label='temporal')
    ax.set_title('(a) Adaptive Perturbation Score (APS) by Branch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('APS Score', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图1] APS 分数曲线已保存为 {save_path}")


def plot_gamma_curve(epochs, gamma, save_path):
    """图2：余弦相似度 γ_m 曲线（MDPS 核心）"""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(epochs, gamma['vision'], color='#d62728', marker='o',
            linewidth=2, markersize=5, label='vision')
    ax.plot(epochs, gamma['text'], color='#1f77b4', marker='s',
            linewidth=2, markersize=5, label='text')
    ax.plot(epochs, gamma['temporal'], color='#2ca02c', marker='^',
            linewidth=2, markersize=5, label='temporal')
    ax.set_title('(b) Gradient Alignment γ_m (MDPS) by Branch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cosine Similarity γ_m', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图2] γ_m 曲线已保存为 {save_path}")


def plot_dominant_modality(dominant_history, train_epochs, save_path):
    """图3：主导模态变化"""
    modal_map = {'vision': 0, 'text': 1, 'temporal': 2}
    reverse_map = {0: 'vision', 1: 'text', 2: 'temporal'}
    
    # 按 epoch 取多数投票
    epoch_dominant = []
    n = len(dominant_history)
    epoch_steps = n / train_epochs
    for i in range(train_epochs):
        start = int(round(i * epoch_steps))
        end = int(round((i + 1) * epoch_steps))
        end = min(end, n)
        if end > start:
            votes = [modal_map[dominant_history[j]] for j in range(start, end)]
            most_common = max(set(votes), key=votes.count)
            epoch_dominant.append(most_common)
        else:
            epoch_dominant.append(2)
    
    fig, ax = plt.subplots(figsize=(7.5, 3))
    epochs = list(range(1, train_epochs + 1))
    ax.scatter(epochs, epoch_dominant, c=['#d62728' if x==0 else '#1f77b4' if x==1 else '#2ca02c' for x in epoch_dominant],
               s=100, zorder=5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['vision', 'text', 'temporal'])
    ax.set_title('(c) Dominant Modality per Epoch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图3] 主导模态变化已保存为 {save_path}")


if __name__ == '__main__':
    sam_data = load_sam_history(SAM_HISTORY_PATH)

    # ===== 新的 APS 分数 =====
    aps_score = {
        'vision': compute_epoch_avg(sam_data['aps_score']['vision'], train_epochs),
        'text': compute_epoch_avg(sam_data['aps_score']['text'], train_epochs),
        'temporal': compute_epoch_avg(sam_data['aps_score']['temporal'], train_epochs),
    }

    # ===== 新的余弦相似度 γ_m =====
    gamma = {
        'vision': compute_epoch_avg(sam_data['gamma']['vision'], train_epochs),
        'text': compute_epoch_avg(sam_data['gamma']['text'], train_epochs),
        'temporal': compute_epoch_avg(sam_data['gamma']['temporal'], train_epochs),
    }

    epochs = list(range(1, train_epochs + 1))

    plot_aps_score_curve(epochs, aps_score, 'aps_score_curve.png')
    plot_gamma_curve(epochs, gamma, 'gamma_curve.png')
    plot_dominant_modality(sam_data['dominant'], train_epochs, 'dominant_modality_curve.png')



