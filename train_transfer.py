from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel
from config import image_size, categories
import matplotlib.pyplot as plt
import time

# Your code should change these values based on your choice of dataset for the transfer task
# -------------
input_shape = (image_size[0], image_size[1], 3)
categories_count = 2 # For dog/cat classification
# -------------

models = {
    'transfered_model': TransferedModel,
    'random_model': RandomModel,
}

def plot_history_diff(random_hist, transfered_hist): # Renamed for clarity
    val_acc_random = random_hist.history['val_accuracy']
    val_acc_transfered = transfered_hist.history['val_accuracy']

    epochs_random = range(1, len(val_acc_random) + 1)
    epochs_transfered = range(1, len(val_acc_transfered) + 1) # Corrected to use transfered_hist length
    assert epochs_random == epochs_transfered, "The two models have been tried with different epochs"

    plt.figure(figsize = (24, 6))
    plt.plot(epochs_random, val_acc_random, 'b', label = 'Random Model Accuracy') # Updated label
    plt.plot(epochs_transfered, val_acc_transfered, 'r', label = 'Transfered Model Accuracy') # Updated label
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    # Your code should change the number of epochs
    epochs = 20 # Increased epochs for better comparison
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_transfer_datasets()
    histories = []
    for name, model_class in models.items():
        print('* Training {} for {} epochs'.format(name, epochs))
        model = model_class(input_shape, categories_count)
        model.print_summary()
        history = model.train_model(train_dataset, validation_dataset, epochs)
        histories.append(history)
        print('* Evaluating {}'.format(name))
        model.evaluate(test_dataset)
        print('* Confusion Matrix for {}'.format(name))
        print(model.get_confusion_matrix(test_dataset))
    assert len(histories) == 2, "The number of trained models is not equal to two"
    plot_history_diff(*histories)
