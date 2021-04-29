import pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from toolz import interleave
from sklearn.model_selection import StratifiedShuffleSplit

import src.constants as cst

SEED = 4
legend_mapping = {211: 0,
                  212: 1,
                  213: 2,
                  214: 3,
                  215: 4,
                  216: 5,
                  217: 6,
                  218: 7,
                  219: 8,
                  221: 9,
                  222: 10,
                  223: 11,
                  230: 12,
                  231: 13,
                  232: 14,
                  233: 15,
                  240: 16,
                  250: 17,
                  290: 18,
                  100: 19,
                  300: 20,
                  500: 21,
                  600: 22}
rdata_fn = [x for x in cst.my_project.rdata_dir.glob('*-db.csv')][0]
df = pd.read_csv(rdata_fn, low_memory=False).dropna()
df = df.replace({"level_2": legend_mapping})

# remove classes with less than x data points
# df[['level_2','POINT_ID']].groupby(['level_2']).agg(['count'])


df_strat = df[["POINT_ID", "level_2"]].drop_duplicates().reset_index(drop=True)
df_strat["isTest"] = 1  # add column to know if row is in test or in train set

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
for train_index, _ in sss.split(df_strat["POINT_ID"], df_strat["level_2"], groups=df_strat["level_2"]):
    df_strat.loc[train_index, "isTest"] = 0
df_m = pd.merge(df, df_strat, how="left", on=["POINT_ID", "level_2"])

# We test that 30% per class has isTest==1
print(df_m[["isTest", "level_2"]].groupby(['level_2']).agg(['mean']))

# We test that the intersection of POINT_ID between train and test is empty

# We now extract the data to create the feature set and test set
d1 = df_m[[col for col in df_m if col.startswith('VV_')]]
d2 = df_m[[col for col in df_m if col.startswith('VH_')]]
X = pd.concat([d1, d2], axis=1)[list(interleave([d1, d2]))]

X_train = X.loc[df_m["isTest"] == 0]
X_val = X.loc[df_m["isTest"] == 1]
y_train = df_m.loc[df_m["isTest"] == 0, ["level_2", "POINT_ID"]]
y_val = df_m.loc[df_m["isTest"] == 1, ["level_2", "POINT_ID"]]

df_train = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1, ignore_index=True)
df_val = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1, ignore_index=True)

df_train.to_csv(cst.my_project.train_dir / "train_dataset_sar.csv", header=False, index=False)
df_val.to_csv(cst.my_project.val_dir / "validation_dataset_sar.csv", header=False, index=False)

# --- Test data
rdata_fn = [x for x in cst.my_project.rdata_dir.glob('*1x1.csv')][0]
df = pd.read_csv(rdata_fn, low_memory=False).dropna()
df = df.replace({"level_2": legend_mapping})
d1 = df[[col for col in df if col.startswith('VV_')]]
d2 = df[[col for col in df if col.startswith('VH_')]]
X_test = pd.concat([d1, d2], axis=1)[list(interleave([d1, d2]))]
y_test = df.loc[:, ["level_2", "POINT_ID"]]

df_test = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1, ignore_index=True)

df_test.to_csv(cst.my_project.test_dir / "test_dataset_sar.csv", header=False, index=False)

# example of bayesian optimization with scikit-optimize
from numpy import mean
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import skopt
from sklearn import metrics
from pycm import *

# prepare data sets
X_train = X_train.values
y_train = y_train.level_2.values
X_val = X_val.values
y_val = y_val.level_2.values

for model_name in ['rf', 'svm']:
    if model_name == 'rf':
        # define the model
        model = RandomForestClassifier(n_jobs=8)
        # define the space of hyperparameters to search
        search_space = [
            skopt.space.Integer(5, 40, name='max_depth'),
            skopt.space.Integer(2, 50, name='max_features'),
            skopt.space.Integer(10, 500, name='n_estimators'),
            skopt.space.Real(0.1, 0.8, name='min_samples_split', prior='uniform')]
    elif model_name == 'svm':
        model = SVC(class_weight='balanced', kernel='rbf', random_state=0)
        # define the space of hyperparameters to search
        search_space = [
            skopt.space.Real(0.001, 10, name='C'),
            skopt.space.Real(0.001, 0.5, name='gamma')]

    # define the function used to evaluate a given configuration
    @use_named_args(search_space)
    def evaluate_model(**params):
        # something
        model.set_params(**params)
        model.fit(X_train, y_train)
        # y_preds = model.predict(X_val)
        # my_cm = ConfusionMatrix(y_val, y_preds, digit=9)
        # accuracy = my_cm.overall_stat['Overall MCC']
        # if accuracy is None:
        #    accuracy=0
        #print(accuracy)
        accuracy = model.score(X_val, y_val)
        # accuracy = metrics.accuracy_score(y_val, pred_val)
        print("done")
        return 1.0 - accuracy



    # perform optimization
    result = gp_minimize(evaluate_model, search_space, n_calls=30, n_jobs=4, n_points=4)
    # summarizing finding:
    print('Best Accuracy: %.3f' % (1.0 - result.fun))
    print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))

    if model_name == 'rf':
        clf = RandomForestClassifier(max_depth=result.x[0], max_features=result.x[1],
                                     n_estimators=result.x[2], min_samples_split=result.x[3],
                                     random_state=0, n_jobs=8)
    elif model_name == 'svm':
        clf = SVC(class_weight='balanced', kernel='rbf', random_state=0, C=result.x[0], gamma=result.x[1])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    conf_file = str(cst.root_dir) + f'/{model_name}/confMatrix-{model_name}'  # + '.csv'
    # Level 2
    cm = ConfusionMatrix(y_test.level_2.values, y_pred, digit=9)
    cm.save_obj(conf_file + 'l2')

    # Level 1
    cm1 = ConfusionMatrix(cst.convert_from_class(y_test.level_2.values, cst.class2subgroup),
                          cst.convert_from_class(y_pred, cst.class2subgroup),
                          digit=9)
    cm1.save_obj(conf_file + 'l1')

    # Level 0
    cm0 = ConfusionMatrix(cst.convert_from_class(y_test.level_2.values, cst.class2group),
                          cst.convert_from_class(y_pred, cst.class2group),
                          digit=9)
    cm0.save_obj(conf_file + 'l0')


# EOF