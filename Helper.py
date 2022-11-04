import matplotlib.pyplot as plt

class Helper(object):
    # merge new history objects with older ones
    def append_history(losses, val_losses, accuracy, val_accuracy, history):
        losses = losses + history.history["loss"]
        val_losses = val_losses + history.history["val_loss"]
        accuracy = accuracy + history.history["binary_accuracy"]
        val_accuracy = val_accuracy + history.history["val_binary_accuracy"]
        return losses, val_losses, accuracy, val_accuracy

    def plot_history(losses, val_losses, accuracies, val_accuracies):
        plt.plot(losses)
        plt.plot(val_losses)
        plt.legend(["train_loss", "val_loss"])
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.show()

        plt.plot(accuracies)
        plt.plot(val_accuracies)
        plt.legend(["train_accuracy", "val_accuracy"])
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()