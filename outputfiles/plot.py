import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_val_test_predictions(df_val_, df_test_, av_rmse_val_, r2s_val_, av_rmse_test_, r2s_test_,
                              xlabels_, ylabels_, filename_val='', filename_test=''):
    axes_min = np.floor(np.min(df_val_[:, 0]))
    axes_max = np.ceil(np.max(df_val_[:, 0]))

    plt.plot([axes_min, 5], [axes_min, 5], '-', color='black')
    plt.plot(df_val_[:, 1], df_val_[:, 0], '.')
    plt.title(f'RMSE: {np.round(av_rmse_val_, 4)} - R^2 = {np.round(np.mean(r2s_val_), 4)}')

    plt.xlabel(xlabels_)
    plt.ylabel(ylabels_)
    plt.xlim(axes_min, axes_max)
    plt.ylim(axes_min, axes_max)

    if filename_val != '':
        plt.savefig(filename_val)
    plt.close()

    plt.plot([axes_min, axes_max], [axes_min, axes_max], '--', color='black')
    plt.plot(df_test_[:, 1], df_test_[:, 0], '.', color='orange')
    plt.title(f'RMSE: {np.round(av_rmse_test_, 4)} - R^2 = {np.round(np.mean(r2s_test_), 4)}')

    plt.xlabel(xlabels_)
    plt.ylabel(ylabels_)
    plt.xlim(axes_min, axes_max)
    plt.ylim(axes_min, axes_max)

    if filename_test != '':
        plt.savefig(filename_test)
    plt.close()


def plot_val_test_predictions_with_details(df_val_, df_test_, av_rmse_val_, r2s_val_, av_rmse_test_, r2s_test_,
                              xlabels_, ylabels_, df_details_, filename_val='', filename_test=''):
    axes_min = np.floor(np.min(df_val_[:, 0]))
    axes_max = np.ceil(np.max(df_val_[:, 0]))
    au = df_details_[:,0]
    years = df_details_[:,1]
    plt.plot([axes_min, 5], [axes_min, 5], '-', color='black')

    plt.plot(df_val_[:, 1], df_val_[:, 0], '.')
    plt.title(f'RMSE: {np.round(av_rmse_val_, 4)} - R^2 = {np.round(np.mean(r2s_val_), 4)}')

    plt.xlabel(xlabels_)
    plt.ylabel(ylabels_)
    plt.xlim(axes_min, axes_max)
    plt.ylim(axes_min, axes_max)

    if filename_val != '':
        plt.savefig(filename_val)
    plt.close()

    plt.plot([axes_min, axes_max], [axes_min, axes_max], '--', color='black')
    #plt.plot(df_test_[:, 1], df_test_[:, 0], '.', color='orange')
    clrplt = sns.color_palette("husl", len(np.unique(au)))
    g = sns.scatterplot(x=df_test_[:, 1], y=df_test_[:, 0], hue=au, style=years,
                     palette=clrplt, legend='full')#palette="Spectral", legend='full')
    # resize to accomodat legend
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    g.legend(loc='upper left', bbox_to_anchor=(1, 1.1), ncol=2)

    plt.title(f'RMSE: {np.round(av_rmse_test_, 4)} - R^2 = {np.round(np.mean(r2s_test_), 4)}')

    plt.xlabel(xlabels_)
    plt.ylabel(ylabels_)
    plt.xlim(axes_min, axes_max)
    plt.ylim(axes_min, axes_max)

    if filename_test != '':
        plt.savefig(filename_test)
    plt.close()

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


def plotHisto(fn, unit, var):
    pd.set_option('display.max_columns', None)
    desired_width = 520
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)
    df = pd.read_csv(fn)
    binDict = {
        'NDVI': {'min': 0.05, 'range': 0.85, 'n': 64},
        'rad': {'min': 40000, 'range': 280000, 'n': 64},
        'rainfall': {'min': 0, 'range': 100, 'n': 64},
        'temperature': {'min': -5, 'range': 50, 'n': 64}
    }

    df = df[df['adm1_name']==unit]
    df = df[df['variable_name'] == var]
    df['dates'] = pd.to_datetime(df['dekad'], format='%Y%m%d')
    df = df.sort_values(by='dates')
    xValues = df['dates'].tolist()
    binValues = np.linspace(binDict[var]['min'], binDict[var]['min'] + binDict[var]['range'], num=binDict[var]['n']+1, endpoint=True)
    yValues = binValues[0:-1]+(binValues[1]-binValues[0])/2
    histo_cols = [col for col in df.columns if 'cls_cnt' in col]
    histo = df[histo_cols].to_numpy().transpose()#np.flip(df[histo_cols].to_numpy().transpose(), axis=0)
    fig = plt.figure()
    plt.pcolormesh(xValues, yValues, histo, vmin=histo.min(), vmax=histo.max(), shading='auto')
    #plt.xlim(left=pd.to_datetime('2002-01-01').to_numpy(), right=pd.to_datetime('2002-12-31').to_numpy())
    # plot the weighted mean to be sure it is correct
    if False:
        wmean = np.zeros(histo.shape[1])
        for i in range(histo.shape[1]):
            wmean[i] = np.sum(histo[:, i] * yValues)/np.sum(histo[:, i])
        plt.plot(xValues, wmean, color='red', linewidth=0.5)
    plt.title(var +' - ' + unit)
    plt.show()
    print('debug')

def plot_2D_inputs_by_region(hist, variables, title, fig_name=None, _figsize=(16.5, 5)):

    fig, axs = plt.subplots(1, 4, figsize=_figsize)
    cmaps = ['Greens',  'Purples', 'Blues', 'Reds']
    for col in range(len(cmaps)):
        ax = axs[col]
        plt.sca(ax)
        pcm = ax.imshow(np.flipud(hist[:, :, col]), cmap=cmaps[col])
        ax.set_title(variables[col])
        fig.colorbar(pcm, ax=ax)
        #plt.plot([3, 3], [0, 64], color='black')
        #plt.plot([3 + 9 * 3, 3 + 9 * 3], [0, 64], color='black')

    plt.suptitle(title)
    plt.tight_layout()

    if fig_name is not None:
        plt.savefig(fig_name)

def plot_1D_inputs_by_region(row, fig_name=None, _figsize=(16.5, 5)):


    vars = ['NDVI', 'rad', 'rainfall', 'temperature']
    fig, axs = plt.subplots(1, 4, figsize=_figsize)
    clr = ['Green', 'Purple', 'Blue', 'Red']
    for col in range(len(clr)):
        ax = axs[col]
        plt.sca(ax)
        y = row.filter(like=vars[col], axis=0).values
        plt.plot(y, color=clr[col])

        ax.set_title(vars[col])

    plt.tight_layout()

    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()

#

