import os
from pathlib import Path
import getpass
import sys
import shutil

# set repository path dependant on username, add new users here:
if getpass.getuser() == 'franc':
    root_dir = Path('C:/Users/franc/OneDrive/Documents/projects/crop_classification')
    git_dir = Path(r'C:/Users/franc/OneDrive/Documents/git/temporalCNN')
elif getpass.getuser() == 'waldnfr':
    root_dir = Path('C:/Users/waldnfr/Documents/projects/crop_classification')
    git_dir = Path(r'C:/Users/waldnfr/Documents/git')
elif getpass.getuser() == 'ml4castproc':
    root_dir = Path(r"/eos/jeodpp/data/projects/ML4CAST/p2s2")
    git_dir = Path(r'/eos/jeodpp/data/projects/ML4CAST/1DcropID')

for i in [str(root_dir), str(git_dir)]:
    sys.path.insert(0, i)


class Project(object):
    def __init__(self, root, raw_data_dir):
        self.root_dir = root
        self.rdata_dir = raw_data_dir
        self.data_dir = root / 'data'
        self.train_dir = self.data_dir / 'training'
        self.val_dir = self.data_dir / 'validation'
        self.test_dir = self.data_dir / 'testing'
        self.meta_dir = self.data_dir / 'meta'
        self.params_dir = self.data_dir / 'params'
        self.figs_dir = self.root_dir / 'figures'

    def create(self, raw_data=[]):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        def create_ts_lbl(fp):
            fp.mkdir(parents=True, exist_ok=True)
            subdirs = ['timeseries', 'labels']
            for subdir in subdirs:
                _subdir = fp / subdir
                _subdir.mkdir(parents=True, exist_ok=True)

        for _fp in [self.train_dir, self.val_dir, self.test_dir, self.meta_dir, self.params_dir, self.figs_dir]:
            _fp.mkdir(parents=True, exist_ok=True)


case = "tempcnn"
my_project = Project(root_dir / case, root_dir / "raw_data")
my_project.create()

class2subgroup = {0: 0,
                  1: 0,
                  2: 0,
                  3: 0,
                  4: 0,
                  5: 0,
                  6: 0,
                  7: 0,
                  8: 0,
                  9: 1,
                  10: 1,
                  11: 1,
                  12: 2,
                  13: 2,
                  14: 2,
                  15: 2,
                  16: 3,
                  17: 4,
                  18: 5,
                  19: 6,
                  20: 7,
                  21: 8,
                  22: 9}

class2group = {0: 0,
               1: 0,
               2: 0,
               3: 0,
               4: 0,
               5: 0,
               6: 0,
               7: 0,
               8: 0,
               9: 0,
               10: 0,
               11: 0,
               12: 0,
               13: 0,
               14: 0,
               15: 0,
               16: 0,
               17: 0,
               18: 0,
               19: 1,
               20: 1,
               21: 1,
               22: 1}

def convert_from_class(my_array, my_dic):
    out = [my_dic[x] for x in my_array]
    return out