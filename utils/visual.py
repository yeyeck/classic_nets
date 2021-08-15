import matplotlib.pyplot as plt
import os


def plot_loss(train_loss, val_loss, save_to):
    x = list(range(len(train_loss)))
    y1 = train_loss
    y2 = val_loss
    plt.figure()
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x, y1, color='b', label='train')
    plt.plot(x, y2, color='r', label='val')
    plt.legend()
    plt.savefig(os.path.join(save_to, 'loss.jpg'))
    plt.close()

def plot_lr(lrs, save_to):
    x = list(range(len(lrs)))
    y = lrs
    plt.title('lr scheduler')
    plt.xlabel('epochs')
    plt.ylabel('lr')
    plt.plot(x, y, color='r', label='learning rate')
    plt.legend()
    plt.savefig(os.path.join(save_to, 'lr.jpg'))
    plt.close()


def plot_accuracy(accuracy, save_to):
    x = list(range(len(accuracy)))
    y = accuracy
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy(%)')
    plt.plot(x, y, color='r', label='Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_to, 'accuracy.jpg'))
    plt.close()