import pickle


# with open('/media/hjx/dataset/DDAD/meta_data/info_{}.pkl'.format('train'), 'rb') as f:
#     info = pickle.load(f)
# print(info)

# from __future__ import absolute_import, division, print_function
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
from runer import Runer
from options import MonodepthOptions
from datasets.ddad_dataset import DDADDataset
from torch.utils.data import DataLoader
import torch

def my_collate(self,batch):
    batch_new = {}
    keys_list = list(batch[0].keys())
    special_key_list = ['id', 'match_spatial']

    for key in keys_list: 
        if key not in special_key_list:
            batch_new[key] = [item[key] for item in batch]
            batch_new[key] = torch.cat(batch_new[key], axis=0)
        else:
            batch_new[key] = []
            for item in batch:
                for value in item[key]:
                    batch_new[key].append(value)

    return batch_new

options = MonodepthOptions()
opts = options.parse()
train_dataset = DDADDataset(opts,
            opts.height, opts.width,
            opts.frame_ids, 4, is_train=True)
train_loader = DataLoader(
            train_dataset, 2, collate_fn=my_collate,
            num_workers=4, pin_memory=True, drop_last=True, #sampler=train_sampler
            )

for batch in train_dataset:
    print('111')