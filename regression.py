from easytorch import ETDataset, ETTrainer, ETDataHandle, Prf1a
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
            self.labels = pd.read_csv(dt['label_csv_path'])

        label = self.labels[self.labels['subjectkey'] == file][dt['labels_column']].values[0]
        self.indices.append([dataset_name, file, label])

    def __getitem__(self, ix):
        dataset, file, label = self.indices[ix]
        dt = self.dataspecs[dataset]

        nii_file = glob.glob(f"{dt['data_dir']}{os.sep}{file.replace('_', '')}/**/Sm6mwc1pT1.nii", recursive=True)[0]
        arr = np.array(ni.load(nii_file).dataobj)[None, ...]
        if self.mode == 'train':
            arr = self.transform(arr)

        label = int(label > 60)

        return {'input': arr, 'label': label}


class ABCDTrainer(ETTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.pred_results = []

    def _init_nn_model(self):
        self.nn['model'] = VGG(
            num_channels=self.args.get('input_channel', 1),
            num_classes=self.args.get('num_class', 1),
            b_mul=self.args.get('b_mul', 1)
        )

    def iteration(self, batch) -> dict:
        input = batch['input'].to(self.device['gpu']).float()
        label = batch['label'].to(self.device['gpu']).long()
        out = self.nn['model'](input)
        loss = F.cross_entropy(out, label)
        out = F.softmax(out, 1)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, label)

        avg = self.new_averages()
        avg.add(loss.item(), len(input))

        return {'loss': loss, 'averages': avg, 'pred': out[:, 1], 'metrics': sc, 'label': label}

    def init_experiment_cache(self):
        self.cache.update(monitor_metric='average', metric_direction='minimize')
        self.cache.update(log_header='MSE')

    def save_predictions(self, dataset, its) -> dict:
        pred = its['pred'].detach().cpu().numpy().tolist()
        label = its['label'].detach().cpu().numpy().tolist()
        self.pred_results += list(zip(pred, label))
        return {}

    def inference(self, mode='test', save_predictions=True, datasets: list = None, distributed=False):
        result = super().inference(mode, save_predictions, datasets, distributed)
        df = pd.DataFrame(self.pred_results, columns=['pred', 'true'])
        df['MAE'] = (df['pred'] - df['true']).abs()
        df['MSE'] = (df['pred'] - df['true']) ** 2
        df = df.append(df[['MAE', 'MSE']].mean(), ignore_index=True).fillna('Total')
        with open(self.cache['log_dir'] + os.sep + self.cache['experiment_id'] + '_predictions.csv', 'w') as f:
            f.write(df.to_csv(index=False))
        return result

    def new_metrics(self):
        return Prf1a()


class ABCDDataHandle(ETDataHandle):
    def _list_files(self, dspec) -> list:
        valid_subjects = pd.read_csv(dspec['valid_subjects_path'])['filename'].tolist()
        files = pd.read_csv(dspec['label_csv_path'])['subjectkey'].tolist()
        # files = ['NDAR_INV0A4P0LWM', 'NDAR_INV0A4ZDYNL', 'NDAR_INV0A6WVRZY']
        valid_files = [f for f in files if f.replace('_', '') in valid_subjects]
        return valid_files

    def get_loader(self, handle_key='', distributed=False, use_unpadded_sampler=False, **kw):
        sampler = None
        if handle_key == 'train':
            """Oversample skewed dataset by using Weighted Sampler"""
            kw['dataset'].indices = sorted(kw['dataset'].indices, key=lambda x: x[2])
            _labels = [a[2] for a in kw['dataset'].indices]
            hist = torch.histc(torch.Tensor(_labels), 10)

            hist_adjusted = hist.clone()
            offset = 0.13
            hist_adjusted[hist < offset * len(_labels)] += offset * len(_labels)
            hist_sum = hist_adjusted.sum().item()

            probs = []
            for i in range(len(hist)):
                for _ in range(int(hist[i])):
                    probs.append((hist_sum - hist_adjusted[i].item()) / hist_sum)

            probs = np.array(probs) / sum(probs)

            assert len(probs) == len(
                _labels), f"Sampling Weights should be equal: probs {len(probs)}, labels {len(_labels)}"
            sampler = WeightedRandomSampler(probs, len(_labels))
        return super().get_loader(handle_key, distributed, use_unpadded_sampler, sampler=sampler, **kw)
