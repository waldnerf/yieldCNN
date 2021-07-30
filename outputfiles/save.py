#!/usr/bin/python

""" 
	Saving output files
"""

import numpy as np
import pandas as pd
from pathlib import Path
import shutil


# -----------------------------------------------------------------------
def save_best_model(path, pattern):
    """
	"""
    best_dir = path / 'best_model'
    best_dir.mkdir(parents=True, exist_ok=True)
    fns_select = path.glob(f'*_{pattern}_*')
    for i in fns_select:
        print(i)
        shutil.copy(i, str(best_dir))


# -----------------------------------------------------------------------
def rm_tree(pth):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)


# -----------------------------------------------------------------------
def saveLossAcc(model_hist, filename):
    """
		Save all the accuracy measures into a csv file
		INPUT:
			- model_hist: all accuracy measures
			- filename: csv file where to store the dictionary 
				(erase the file it does already exist)
				8 significant digits after the decimal point
	"""
    f = open(filename, 'w')
    for key in model_hist.keys():
        line = key + ',' + ','.join(map(str, model_hist[key])) + '\n'
        f.write(line)
    f.close()


# -----------------------------------------------------------------------
def saveMatrix(mat, filename, label):
    """
		Save numpy array into a csv file
		INPUT:
			- mat: numpy array
			- filename: csv file where to store the mat array 
				(erase the file it does already exist)
				8 significant digits after the decimal point
			- label: name of the columns
	"""
    df = pd.DataFrame(mat, columns=label)
    df.to_csv(filename)


# -----------------------------------------------------------------------
def readMatrix(filename):
    """
		Save numpy array into a csv file
		INPUT:
			- filename: csv file where to store the mat array 
				(erase the file it does already exist)
				8 significant digits after the decimal point
		OUTPUT:
			- res_mat: numpy array
	"""
    mat = pd.read_csv(filename)
    return mat.values


# -----------------------------------------------------------------------
def write_predictions_csv(test_file, p_test):
    """
		 Writing the predictions p_test in test_file
		 INPUT:
			-test_file: csv file where to store the results
			-p_test: predictions 
				(either predicted class 
					or class probability distribution outputing by the Softmax layer)
	"""
    print("len(p_test.shape)", len(p_test.shape))
    if len(p_test.shape) == 1:  # -- saving class only [integer]
        np.savetxt(test_file, p_test.astype(int), delimiter=',', fmt='%i')
    else:  # saving proba [float]
        np.savetxt(test_file, p_test, delimiter=',', fmt='%1.6f')

# EOF
