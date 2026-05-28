from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

# ====================================
# PATHS TO TWO LOG FOLDERS
# ====================================
train_log_dir = "/mnt/storage/divya/exam/graformer_workspace/dgpt124m_checkpoints/tensorboard/train_inner"
val_log_dir = "/mnt/storage/divya/exam/graformer_workspace/dgpt124m_checkpoints/tensorboard/valid"

# TensorBoard scalar tag
tag = "loss"

# ====================================
# LOAD TRAIN EVENTS
# ====================================
train_acc = EventAccumulator(train_log_dir)
train_acc.Reload()

train_events = train_acc.Scalars(tag)

train_steps = [x.step for x in train_events]
train_loss = [x.value for x in train_events]

# ====================================
# LOAD VALID EVENTS
# ====================================
val_acc = EventAccumulator(val_log_dir)
val_acc.Reload()

val_events = val_acc.Scalars(tag)

val_steps = [x.step for x in val_events]
val_loss = [x.value for x in val_events]

# ====================================
# MIN VALUES
# ====================================
train_min = np.min(train_loss)
val_min = np.min(val_loss)

# ====================================
# PLOT
# ====================================
plt.figure(figsize=(10, 6))

plt.plot(
    train_steps,
    train_loss,
    linewidth=3,
    label=f"Training Loss (Min: {train_min:.2f})"
)

plt.plot(
    val_steps,
    val_loss,
    linestyle="--",
    linewidth=3,
    label=f"Validation Loss (Min: {val_min:.2f})"
)

plt.xlabel("Steps", fontsize=16)
plt.ylabel("Loss", fontsize=16)

plt.title("Training Loss vs Validation Loss", fontsize=20)

plt.legend(fontsize=14)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("train_vs_val_loss_nmt1.png", dpi=300)

plt.show()
