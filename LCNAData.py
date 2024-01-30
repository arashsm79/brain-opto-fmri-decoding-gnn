import os.path as osp
import pandas as pd

import torch
from torch_geometric.data import Dataset, download_url


class LCNAData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_labels = pd.read_csv(osp.join(root, 'data_labels.csv'))

    def len(self):
        return len(self.data_labels)

    def get(self, idx):
        subject_id = self.data_labels.iloc[idx]['subject_id']
        protocol_label = self.data_labels.iloc[idx]['label']
        sample_id = self.data_labels.iloc[idx]['sample_id']
        stim_id = self.data_labels.iloc[idx]['stim_id']
        data = torch.load(osp.join(self.root, str(subject_id), f'sub{subject_id}_prot{protocol_label}_stim{stim_id}_sample{sample_id}.pt'))
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.x[data.x == float('inf')] = 0
        data.edge_attr[data.edge_attr == float('inf')] = 0
        return data
