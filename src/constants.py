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
    elif getpass.getuser() == 'waldnfr':
        root_dir = Path('C:/Users/waldnfr/Documents/projects/leanyf')
        git_dir = Path(r'C:/Users/waldnfr/Documents/git')
else:
    if getpass.getuser() == 'waldnfr':
        root_dir = Path(r"/home/waldnfr/data/leanyf")
        git_dir = Path(r'/home/waldnfr/git')

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



