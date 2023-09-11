import csv
import time
from lightning.pytorch.callbacks import Callback

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

    def on_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_epoch_end(self, trainer, pl_module):
        # Calculate epoch duration
        duration = time.time() - self.start_time

         # Get the desired metrics
        train_loss = trainer.callback_metrics.get("train_loss", None)
        val_loss = trainer.callback_metrics.get("val_loss", None)
        train_acc = trainer.callback_metrics.get("train_accuracy", None)
        val_acc = trainer.callback_metrics.get("val_accuracy", None)

        # Print the metrics
        print(f"Epoch {trainer.current_epoch}:")
        print(f"  Train Loss: {train_loss}, Train Accuracy: {train_acc}")
        print(f"  Val Loss: {val_loss}, Val Accuracy: {val_acc}")
        print(f"  Epoch Time: {duration} seconds")

        # Write to the CSV file
        mode = "a" if trainer.current_epoch > 0 else "w"
        with open(self.file_path, mode, newline='') as file:
            writer = csv.writer(file)
            if trainer.current_epoch == 0:
                writer.writerow(["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "epoch_time(s)"])
            writer.writerow([trainer.current_epoch, train_loss, val_loss, train_acc, val_acc, duration])