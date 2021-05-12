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
    root_dir = Path('C:/Users/waldnfr/Documents/projects/leanyf')
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
        self.meta_dir = self.data_dir / 'meta'
        self.params_dir = self.data_dir / 'params'
        self.figs_dir = self.root_dir / 'figures'

    def create(self, raw_data=[]):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for _fp in [self.meta_dir, self.params_dir, self.figs_dir]:
            _fp.mkdir(parents=True, exist_ok=True)


my_project = Project(root_dir, root_dir / "raw_data")
my_project.create()

target = 'Algeria'

if target == 'Algeria':
    step_dic = {'11-01': 1, '11-11': 2, '11-21': 3, '12-01': 4, '12-11': 5, '12-21': 6, '01-01': 7, '01-11': 8,
                '01-21': 9,
                '02-01': 10, '02-11': 11, '02-21': 12, '03-01': 13, '03-11': 14, '03-21': 15, '04-01': 16, '04-11': 17,
                '04-21': 18, '05-01': 19, '05-11': 20, '05-21': 21, '06-01': 22, '06-11': 23, '06-21': 24
                }
    month_sos = 11
