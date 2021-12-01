import os
from regression import ABCDTrainer, ABCDDataHandle, ABCDDataset
from easytorch import EasyTorch

ABCD = {
    'name': 'ABC3D',
    'data_dir': 'ABCD' + os.sep + 'volumes',
    'label_dir': 'ABCD' + os.sep + 'labels',
    'labels_column': 'cbcl_scr_syn_attention_t'
}

if __name__ == "__main__":
    runner = EasyTorch([ABCD])
    runner.run(ABCDTrainer, ABCDDataset, ABCDDataHandle)
