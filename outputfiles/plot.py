import matplotlib.pylab as plt
import numpy as np

def plot_predictions(y, preds, title='', fn=''):
    plt.figure()
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '-r', color='black')
    plt.plot(preds[:, 0], y[:, 0], '.', label=f'Crop 1')
    plt.plot(preds[:, 1], y[:, 1], '.', label=f'Crop 2')
    plt.plot(preds[:, 2], y[:, 2], '.', label=f'Crop 3')
    plt.title(title)
    plt.legend()

    if fn != '':
        plt.savefig(fn)
