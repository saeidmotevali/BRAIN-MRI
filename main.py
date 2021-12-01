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

ABCD = {
    'name': 'ABC3D',
    'data_dir': 'ABCD' + os.sep + 'volumes',
    'label_dir': 'ABCD' + os.sep + 'labels',
    'labels_column': 'cbcl_scr_syn_attention_t'
}

"""
python main.py -ph train -b 32 -e 151 --b-mul 4 -lr 0.005 -nf 10
"""
if __name__ == "__main__":

    runner = EasyTorch([ABCD], args=args)
    runner.run(ABCDTrainer, ABCDDataset, ABCDDataHandle)
