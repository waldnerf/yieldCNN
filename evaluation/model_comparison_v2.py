import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os.path


import mysrc.constants as cst


def plot_accuracy_vs_time(df_, my_colors_, x_labels, filename=''):
    fig, axs = plt.subplots(df_.Crop.unique().shape[0], figsize=(8, 20))

    for i in range(df_.Crop.unique().shape[0]):
        df_i = df_.loc[df_.Crop == df_.Crop.unique()[i],].copy()
        for j, model_type in enumerate(df_i.Estimator.unique()):
            axs[i].plot(df_i.loc[df_i.Estimator == model_type, 'lead_time'].values,
                        df_i.loc[df_i.Estimator == model_type, 'rRMSE_p'].values,
                        color=my_colors_[j], label=model_type)
            axs[i].set_title(df_.Crop.unique()[i], fontweight="bold")
            axs[i].set_ylabel('rRMSE (%)')
            axs[i].set_ylim([9, 51])
            axs[i].set_yticks(range(10, 51, 10))
            axs[i].set_xticks(df_i.lead_time.unique())
            axs[i].set_xticklabels(x_labels)
            if i == (df_i.Estimator.unique().shape[0] - 1):
                axs[i].set_xlabel('Forcast date')
                axs[i].legend(loc="lower left", title="", frameon=False)
    plt.subplots_adjust(hspace=0.3)
    #plt.show()
    if filename != '':
        plt.savefig(filename, dpi=450)

target_var = 'Yield'
# -- Read in results
# ML and simple benchmarks
rdata_dir = Path(cst.root_dir, 'raw_data')
fn_benchmark = rdata_dir / r'all_model_output.csv'#best_ML_benchnarks.csv'
df_bench = pd.read_csv(fn_benchmark)
df_bench = df_bench.loc[:, ['lead_time', 'Crop', 'Estimator', 'rRMSE_p']].copy()
ML_selector = [False if x in ['PeakNDVI', 'Null_model'] else True for x in df_bench.Estimator]
df_bench.loc[ML_selector, 'Estimator'] = 'Machine Learning'
df_bench = df_bench.groupby(['lead_time', 'Crop', 'Estimator']).min().reset_index(level=[0, 1, 2], drop=False)
df_bench.loc[df_bench.Estimator == 'Null_model', 'Estimator'] = 'Null model'
df_bench.loc[df_bench.Estimator == 'PeakNDVI', 'Estimator'] = 'Peak NDVI'



# -- Plot Best 2D CNN vs best benchmarks
x_tick_labels = ['Dec 1', 'Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1', 'Jul 1']
my_colors = ['#78b6fc', '#a9a9a9', '#ffc000' ]# '#034da2']

if os.path.exists(cst.root_dir / f"data/model_evaluation_2DCNN.csv"):
    df_2D = pd.read_csv(cst.root_dir / f"data/model_evaluation_2DCNN.csv")
if os.path.exists(cst.root_dir / f"data/model_evaluation_1DCNN.csv"):
    df_1D = pd.read_csv(cst.root_dir / f"data/model_evaluation_1DCNN.csv")

for crop_name in df_bench['Crop'].unique():
    fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
    #benchmark
    df_i = df_bench.loc[df_bench.Crop == crop_name].copy()
    mdl = 'Null model'
    axs.plot(df_i.loc[df_i.Estimator == mdl, 'lead_time'].values, df_i.loc[df_i.Estimator == mdl, 'rRMSE_p'].values,
             color='grey', linewidth=1, marker='o', label=mdl)
    mdl = 'Peak NDVI'
    axs.plot(df_i.loc[df_i.Estimator == mdl, 'lead_time'].values, df_i.loc[df_i.Estimator == mdl, 'rRMSE_p'].values,
             color='red', linewidth=1, marker='o', label=mdl)
    mdl = 'Machine Learning'
    axs.plot(df_i.loc[df_i.Estimator == mdl, 'lead_time'].values, df_i.loc[df_i.Estimator == mdl, 'rRMSE_p'].values,
             color='blue', linewidth=1, marker='o', label=mdl)
    # plot 2d results if present
    if os.path.exists(cst.root_dir / f"data/model_evaluation_2DCNN.csv"):
        crop_ind = cst.crop_name_ind_dict[crop_name]

        df = df_2D.loc[df_2D.targetVar == target_var.lower()].copy()
        if crop_name in df.Crop.unique():
            dfc = df.loc[df.Crop == crop_name].copy()
            colors = ['lime','darkgreen','indigo','magenta']
            for j, model_type in enumerate(dfc.Estimator.unique()):
                axs.plot(dfc.loc[dfc.Estimator == model_type, 'lead_time'].values, dfc.loc[dfc.Estimator == model_type, 'rRMSE_p'].values, color=colors[j], linewidth=1, marker='o', label=model_type)



    #aggiustare grafico come ML e salvare!!
    axs.set_ylim(0, 50)
    axs.set_ylabel('rRMSEp (%)')
    axs.set_xticks(df_i.lead_time.unique())
    axs.set_xticklabels(x_tick_labels)
    axs.set_xlabel('Forecast time')
    # axs.set_title(crop_name + ', ' + y_var, fontsize=12)
    axs.set_title(crop_name, fontsize=12)
    # axs.set_title(axTilte,  fontsize=12)
    # fig.suptitle(crop_name + ', ' + y_var, fontsize=14, fontweight='bold')
    axs.legend(frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")
    #axs.legend(frameon=False, loc='upper right') #, ncol=len(axs.lines)
    fn = cst.root_dir / f'data/ML_performances_{crop_name}.png'
    plt.savefig(fn, dpi=450)
    plt.close()

#OLD CODE OF FRANZ
fn = cst.root_dir / f"data/ML_performances.png"
plot_accuracy_vs_time(df_bench, my_colors, x_tick_labels, fn)
# -- Plot 1D CNN
if False:
    df_1D = pd.read_csv(cst.root_dir / f"data/model_evaluation_1DCNN.csv")
    target_var = 'Yield'
    df_ = df_1D.loc[df_1D.targetVar == target_var.lower()].copy()
    super_title = target_var

    fig, axs = plt.subplots(df_.Crop.unique().shape[0], figsize=(8, 20))

    for i in range(df_.Crop.unique().shape[0]):
        df_i = df_.loc[df_.Crop == df_.Crop.unique()[i],].copy()
        for j, model_type in enumerate(df_i.Estimator.unique()):
            axs[i].plot(df_i.loc[df_i.Estimator == model_type, 'lead_time'].values,
                        df_i.loc[df_i.Estimator == model_type, 'rRMSE_p'].values,
                        label=model_type)#color=my_colors[j], label=model_type)
            axs[i].set_title(df_.Crop.unique()[i], fontweight="bold")
            axs[i].set_ylabel('rRMSE (%)')
            axs[i].set_ylim([9, 51])
            axs[i].set_yticks(range(10, 51, 10))
            axs[i].set_xticks(df_i.lead_time.unique())
            #axs[i].set_xticklabels(x_tick_labels)
            if i == (df_i.Estimator.unique().shape[0] - 1):
                axs[i].set_xlabel('Forcast date')
                axs[i].legend(loc="lower left", title="", frameon=False)
    plt.subplots_adjust(hspace=0.3, top=0.92)#, bottom=0.1)
    plt.suptitle(super_title, fontsize=15)
    #plt.show()
    filename = cst.root_dir / f"data/1D_performances.png"
    if filename != '':
        plt.savefig(filename, dpi=450)



# Compare 2D CNNs

df_2D = pd.read_csv(cst.root_dir / f"data/model_evaluation_2DCNN.csv")

target_var = 'Yield'
df_ = df_2D.loc[df_2D.targetVar == target_var.lower()].copy()
super_title = target_var

fig, axs = plt.subplots(df_.Crop.unique().shape[0], figsize=(8, 20))

for i in range(df_.Crop.unique().shape[0]):
    df_i = df_.loc[df_.Crop == df_.Crop.unique()[i],].copy()
    for j, model_type in enumerate(df_i.Estimator.unique()):
        axs[i].plot(df_i.loc[df_i.Estimator == model_type, 'lead_time'].values,
                    df_i.loc[df_i.Estimator == model_type, 'rRMSE_p'].values,
                    label=model_type)#color=my_colors[j], label=model_type)
        axs[i].set_title(df_.Crop.unique()[i], fontweight="bold")
        axs[i].set_ylabel('rRMSE (%)')
        axs[i].set_ylim([9, 51])
        axs[i].set_yticks(range(10, 51, 10))
        axs[i].set_xticks(df_i.lead_time.unique())
        #axs[i].set_xticklabels(x_tick_labels)
        if i == (df_i.Estimator.unique().shape[0] - 1):
            axs[i].set_xlabel('Forcast date')
            axs[i].legend(loc="lower left", title="", frameon=False)
plt.subplots_adjust(hspace=0.3, top=0.92)#, bottom=0.1)
plt.suptitle(super_title, fontsize=15)
#plt.show()
filename = cst.root_dir / f"data/2D_performances.png"

if filename != '':
    plt.savefig(filename, dpi=450)


