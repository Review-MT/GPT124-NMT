from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

log_dir ="/mnt/storage/divya/exam/graformer_workspace/dgpt124m_checkpoints/tensorboard/train"# "./"   # folder containing event file

event_acc = EventAccumulator(log_dir)
event_acc.Reload()

print("Available tags:")
print(event_acc.Tags()['scalars'])

# Extract training loss
loss_events = event_acc.Scalars("loss")

steps = [x.step for x in loss_events]
loss_values = [x.value for x in loss_events]

# Plot
plt.figure(figsize=(8,5))

plt.plot(steps, loss_values, label="Training Loss")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)

plt.savefig("training_loss_curve.png", dpi=300)
plt.show()
