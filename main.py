import os
from regression import ABCDTrainer, ABCDDataHandle, ABCDDataset
from easytorch import EasyTorch, default_ap
import argparse
import pandas as pd

ap = argparse.ArgumentParser(parents=[default_ap], add_help=False)
ap.add_argument('--num-channel', default=1, type=int, help='Number of input channel.')
ap.add_argument('--num-class', default=1, type=int, help='Number of classes.')
ap.add_argument('--b-mul', default=1, type=int, help='Model scale(higher=bigger model by width).')
args = vars(ap.parse_args())

# ABCD_local = {
#     'name': 'ABCD_local',
#     'data_dir': 'datasets/ABCD/volumes',
#     'label_csv_path': 'datasets/ABCD/labels/abcd_cbcls01L.csv',
#     'valid_subjects_path': 'datasets/ABCD/labels/valid_files_corr.csv',
#     'labels_column': 'cbcl_scr_syn_attention_t'
# }

ABCD = {
    'name': 'ABCD',
    'data_dir': '/data/qneuromark/Data/ABCD/Data_BIDS/Raw_Data/',
    'label_csv_path': '/home/users/smotevalialamoti1/SMLvsDL/abcd_cbcls01L.csv',
    'valid_subjects_path': '/home/users/smotevalialamoti1/SMLvsDL/valid_files_corr_98.csv',
    'labels_column': 'cbcl_scr_syn_attention_t'
}

"""
python main.py -ph train -b 32 -e 151 --b-mul 4 -lr 0.005 -nf 10
"""
if __name__ == "__main__":
    dataloader_args = {'train': {'drop_last': True}}
    runner = EasyTorch([ABCD], args=args, dataloader_args=dataloader_args)
    runner.run(ABCDTrainer, ABCDDataset, ABCDDataHandle)
