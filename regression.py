from easytorch import ETDataset, ETTrainer, ETDataHandle
import torchio as tio
import pandas as pd
import os
from model import VGG
import torch.nn.functional as F
import glob
import nibabel as ni
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class ABCDDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None
        self.transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2))])

    def load_index(self, dataset_name, file):
        dt = self.dataspecs[dataset_name]
        if not self.labels:
            _label_file = [f for f in os.listdir(dt['label_dir']) if f.endswith('.csv')][0]
            self.labels = pd.read_csv(dt['label_dir'] + os.sep + _label_file)

        label = self.labels[self.labels['subjectkey'] == file][dt['labels_column']].values[0]
        self.indices.append([dataset_name, file, label])

    def __getitem__(self, item):
        dataset, file, label = self.indices[item]
        dt = self.dataspecs[dataset]

        nii_file = glob.glob(f"{dt['data_dir']}{os.sep}{file.replace('_', '')}/**/Sm6mwc1pT1.nii", recursive=True)[0]
        arr = np.array(ni.load(nii_file).dataobj)[None, ...]
        if self.mode == 'train':
            arr = self.transform(arr)

        return {'input': arr, 'label': np.array([label], dtype=np.float)}


class ABCDTrainer(ETTrainer):
    def _init_nn_model(self):
        self.nn['model'] = VGG(
            num_channels=self.args.get('input_channel', 1),
            num_classes=self.args.get('num_class', 1),
            b_mul=self.args.get('b_mul', 1)
        )

    def iteration(self, batch) -> dict:
        input = batch['input'].to(self.device['gpu']).float()
        label = batch['label'].to(self.device['gpu']).float()
        print(input.shape)
        out = self.nn['model'](input)
        loss = F.mse_loss(out, label)
        mse = self.new_metrics()
        mse.add(loss.item())
        return {'loss': loss, 'metrics': mse, 'averages': mse}

    def init_experiment_cache(self):
        self.cache.update(monitor_metric='average', metric_direction='minimize')
        self.cache.update(log_header='MSE')


class ABCDDataHandle(ETDataHandle):
    def _list_files(self, dspec) -> list:
        _valid_file = [f for f in os.listdir(dspec['label_dir']) if f.endswith('.txt')][0]
        valid_subjects = [l.strip() for l in open(dspec['label_dir'] + os.sep + _valid_file)]
        files = pd.read_csv(dspec['label_dir'] + os.sep + os.listdir(dspec['label_dir'])[0])['subjectkey'].tolist()

        # files = ['NDAR_INV0A4P0LWM', 'NDAR_INV0A4ZDYNL', 'NDAR_INV0A6WVRZY']
        valid_files = [f for f in files if f.replace('_', '') in valid_subjects]
        return valid_files

    # def get_loader(self, handle_key='', distributed=False, use_unpadded_sampler=False, **kw):
    #     sampler = None
    #     if handle_key == 'train':
    #         """Oversample skewed dataset by using Weighted Sampler"""
    #
    #         _labels = [a[-1] for a in kw['dataset'].indices]
    #         hist = torch.histc(torch.Tensor(_labels), 10)
    #
    #         hist_adjusted = hist.clone()
    #
    #         offset = 0.13
    #         hist_adjusted[hist < offset * len(_labels)] += offset * len(_labels)
    #         hist_sum = hist_adjusted.sum().item()
    #
    #         probs = []
    #         for i in range(len(hist)):
    #             for _ in range(int(hist[i])):
    #                 probs.append((hist_sum - hist_adjusted[i].item()) / hist_sum)
    #
    #         probs = np.array(probs) / sum(probs)
    #
    #         assert len(probs) == len(
    #             _labels), f"Sampling Weights should be equal: probs {len(probs)}, labels {len(_labels)}"
    #         sampler = WeightedRandomSampler(probs, len(_labels))
    #     return super().get_loader(handle_key, distributed, use_unpadded_sampler, sampler=sampler, **kw)
