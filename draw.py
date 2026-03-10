import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

pattern = re.compile(r"train_loss:\s*([\d.]+)\s*val_accuracy:\s*([\d.]+)")

log_files = [str(p) for p in Path("logs").rglob("*.log")]
if not log_files:
    raise FileNotFoundError("logs 及其子目录未找到 .log 文件")
print(log_files)

for log_path in log_files:
    print(f"找到日志文件：{log_path}")

    output_png = str(Path(log_path).with_suffix(".png"))
    if os.path.exists(output_png):
        print(f"跳过（已存在同名 PNG）：{output_png}")
        continue

    train_losses, val_accuracies = [], []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                train_losses.append(float(m.group(1)))
                val_accuracies.append(float(m.group(2)))

    n_train, n_val = len(train_losses), len(val_accuracies)
    if n_train == 0 or n_val == 0:
        print(f"跳过（无有效数据）：{log_path}")
        continue
    if n_train != n_val:
        print(f"警告：轮数不一致（train={n_train}, val={n_val}），以较小者为准")
        cut = min(n_train, n_val)
        train_losses, val_accuracies = train_losses[:cut], val_accuracies[:cut]

    if len(train_losses) < 80:
        print(f"跳过（未训练完成 <100 epochs）：{log_path}，仅 {len(train_losses)} 轮")
        continue

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "o-", label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss over Epochs"); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, "s-", label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs"); plt.grid(True)

    max_acc = max(val_accuracies)
    max_epoch = 1 + val_accuracies.index(max_acc)
    plt.scatter(max_epoch, max_acc, s=50, zorder=5, label=f"Best {max_acc:.3f}")
    plt.text(max_epoch, max_acc, f"{max_acc:.3f}", ha="left", va="bottom", fontsize=9)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"图像已保存为 {output_png}")
    plt.show()
    plt.close()