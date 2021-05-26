import matplotlib.pylab as plt
import numpy as np

def plot_predictions_mo(y, preds, title='', fn=''):
    plt.figure()
    plt.plot(np.linspace(0, 5, 10), np.linspace(0, 5, 10), '-r', color='black')
    plt.plot(preds[:, 0], y[:, 0], '.', label=f'Crop 1')
    plt.plot(preds[:, 1], y[:, 1], '.', label=f'Crop 2')
    plt.plot(preds[:, 2], y[:, 2], '.', label=f'Crop 3')
    plt.xlim(0.0, 5.0)
    plt.ylim(0.0, 5.0)
    plt.xlabel('Predictions (t/ha)')
    plt.ylabel('Observations (t/ha)')
    plt.title(title)
    plt.legend()

    if fn != '':
        plt.savefig(fn)
        plt.close()

def plot_predictions_so(y, preds, title='', fn=''):
    plt.figure()
    plt.plot([0, 5], [0, 5], '--', color='black')
    plt.plot(preds, y, '.', color='orange')
    plt.title(title)

    plt.xlabel('Predictions (t/ha)')
    plt.ylabel('Observations (t/ha)')
    plt.xlim(0.0, 5.0)
    plt.ylim(0.0, 5.0)

    if fn != '':
        plt.savefig(fn)
        plt.close()

#
