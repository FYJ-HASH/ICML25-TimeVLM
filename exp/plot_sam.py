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


def plot_aps_curve(epochs, aps, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(epochs, aps['vision'], color='#d62728', marker='o',
            linewidth=2, markersize=5, label='vision')
    ax.plot(epochs, aps['text'], color='#1f77b4', marker='s',
            linewidth=2, markersize=5, label='text')
    ax.plot(epochs, aps['temporal'], color='#2ca02c', marker='^',
            linewidth=2, markersize=5, label='temporal')
    ax.set_title('(a) APS Perturbation Magnitude by Branch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perturbation Norm', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图1] APS 曲线已保存为 {save_path}")


def plot_mdps_curve(epochs, mdps, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, mdps['vision_multi_norm'], color='#d62728', marker='o',
            linewidth=2, markersize=5, label='vision - fusion grad')
    ax.plot(epochs, mdps['vision_uni_norm'], color='#d62728', marker='s',
            linewidth=2, markersize=5, label='vision - single-modal grad')
    ax.plot(epochs, mdps['text_multi_norm'], color='#1f77b4', marker='o',
            linewidth=2, markersize=5, label='text - fusion grad')
    ax.plot(epochs, mdps['text_uni_norm'], color='#1f77b4', marker='s',
            linewidth=2, markersize=5, label='text - single-modal grad')
    ax.plot(epochs, mdps['temporal_multi_norm'], color='#2ca02c', marker='o',
            linewidth=2, markersize=5, label='temporal - fusion grad')
    ax.plot(epochs, mdps['temporal_uni_norm'], color='#2ca02c', marker='s',
            linewidth=2, markersize=5, label='temporal - single-modal grad')
    ax.set_yscale('log')
    ax.set_title('(b) MDPS Gradient Decomposition by Branch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图2] MDPS 曲线已保存为 {save_path}")


if __name__ == '__main__':
    sam_data = load_sam_history(SAM_HISTORY_PATH)

    aps = {
        'vision': compute_epoch_avg(sam_data['aps']['vision'], train_epochs),
        'text': compute_epoch_avg(sam_data['aps']['text'], train_epochs),
        'temporal': compute_epoch_avg(sam_data['aps']['temporal'], train_epochs),
    }

    mdps = {
        'vision_multi_norm': compute_epoch_avg(sam_data['mdps']['vision_multi_norm'], train_epochs),
        'vision_uni_norm': compute_epoch_avg(sam_data['mdps']['vision_uni_norm'], train_epochs),
        'text_multi_norm': compute_epoch_avg(sam_data['mdps']['text_multi_norm'], train_epochs),
        'text_uni_norm': compute_epoch_avg(sam_data['mdps']['text_uni_norm'], train_epochs),
        'temporal_multi_norm': compute_epoch_avg(sam_data['mdps']['temporal_multi_norm'], train_epochs),
        'temporal_uni_norm': compute_epoch_avg(sam_data['mdps']['temporal_uni_norm'], train_epochs),
    }

    epochs = list(range(1, train_epochs + 1))

    plot_aps_curve(epochs, aps, 'aps_epoch_curve.png')
    plot_mdps_curve(epochs, mdps, 'mdps_epoch_curve.png')


