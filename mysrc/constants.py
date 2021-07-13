import os
from pathlib import Path
import getpass
import sys
import shutil

# set repository path dependant on username, add new users here:
if sys.platform == 'win32':
    if getpass.getuser() == 'franc':
        root_dir = Path('C:/Users/franc/OneDrive/Documents/projects/crop_classification')
        git_dir = Path(r'C:/Users/franc/OneDrive/Documents/git/temporalCNN')
        neptune_project = 'waldnerf/yieldCNN'
    elif getpass.getuser() == 'waldnfr':
        root_dir = Path('C:/Users/waldnfr/Documents/projects/leanyf')
        git_dir = Path(r'C:/Users/waldnfr/Documents/git')
        neptune_project = 'waldnerf/yieldCNN'
    elif getpass.getuser() == 'meronmi':
        root_dir = Path(r'D:/PY_data/leanyf')
        git_dir = Path(r'c:/MM_not_sure_what_for')
        neptune_project = 'MM_not_sure_what_for'
       
else:
    if 'google.colab' in str(get_ipython()):
        print('Running on CoLab')
        root_dir = Path('/content/gdrive/MyDrive/leanyf')
        git_dir = Path('/content/yieldCNN/')
        neptune_project = 'waldnerf/yieldCNN'
    else:
        print('Not running on CoLab')
        if getpass.getuser() == 'waldnfr':
            root_dir = Path(r"/home/waldnfr/data/leanyf")
            git_dir = Path(r'/home/waldnfr/git')
            neptune_project = 'waldnerf/yieldCNN'

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
    step_dic = {'10-01': 1, '10-11': 2, '10-21': 3, '11-01': 4, '11-11': 5, '11-21': 6, '12-01': 7, '12-11': 8,
                '12-21': 9, '01-01': 10, '01-11': 11, '01-21': 12, '02-01': 13, '02-11': 14, '02-21': 15, '03-01': 16,
                '03-11': 17, '03-21': 18, '04-01': 19, '04-11': 20, '04-21': 21, '05-01': 22, '05-11': 23, '05-21': 24
                }
    month_sos = 10

