from utils.datasets.moldesign_dataset import *
import os
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter_sum
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,collate_fn=None, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)


def collate_fn_sparse(batch):
    batch = [one for one in batch if one is not None]
    start_idx_list = []
    protein_feature_list = []
    protein_pos_list = []
    protein_edge_idx_list = []
    # ligand_feature_list = []
    # actions_list = []
    protein_batch_id_list = []
    # ligand_batch_id_list = []
    ligand_smiles_list = []
    protein_filename_batch_list =[]
    
    metrics = []

    for i, one in enumerate(batch):
        start_idx_list.append(one[0])
        protein_feature_list.append(one[1])
        protein_pos_list.append(one[2])
        protein_batch_id_list.append(torch.ones(one[1].shape[0])*i)
        protein_filename_batch_list.append(one[3])
        ligand_smiles_list.append(one[5])
        metrics.append(one[6])
    
    start_idx = torch.tensor(start_idx_list)
    protein_feature = 0
    protein_pos = 0
    protein_batch_id = 0
    protein_edge_idx = 0
    if protein_feature_list:
        protein_feature = torch.cat(protein_feature_list, dim=0).float()
    if protein_pos_list:
        protein_pos = torch.cat(protein_pos_list, dim=0)
    
    if protein_batch_id_list:
        protein_batch_id = torch.cat(protein_batch_id_list, dim=0).long()
  
        N_atoms_protein = scatter_sum(torch.ones_like(protein_batch_id), protein_batch_id)#[8]
        shift_protein = torch.cumsum(torch.cat([torch.zeros([1]).long(), N_atoms_protein]), dim=-1)[:-1]#[8]

    for i, one in enumerate(batch):
        local_edge_idx = one[4]
        batch_edge_idx = local_edge_idx + shift_protein[i]

        protein_edge_idx_list.append(batch_edge_idx)
    if  protein_edge_idx_list:
        protein_edge_idx = torch.cat(protein_edge_idx_list, dim=-1)
   
    return start_idx, protein_feature, protein_pos,  protein_batch_id, protein_edge_idx, protein_filename_batch_list, ligand_smiles_list, metrics#, compose_batch_id

def get_dataset(root="./data/crossdocked_pocket10",split_path = './data/split_by_name.pt',num_workers=8, batch_size=4, mode="train", distributed=True):
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
            RefineData(),
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer
        ])

    dataset = MolDesign_dataset(root=root,
                                split_path = split_path,
                                mode = mode,
                                transform=transform)
    
    if distributed:
        dataset_sample = torch.utils.data.distributed.DistributedSampler(dataset)
    
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_sparse, num_workers=num_workers, drop_last=False, sampler=dataset_sample, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_sparse, num_workers=num_workers, drop_last=True, pin_memory=True)
    
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataset()
    
    for data in dataloader:
        print(data)