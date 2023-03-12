from torch.utils.data import Dataset
from .pl import PocketLigandPairDataset
import torch
from torch.utils.data import Subset
import copy
import random
import torch.nn.functional as F
from itertools import compress
from torch_geometric.transforms import Compose
import numpy as np
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.nn.pool import knn_graph, radius, knn
from rdkit import  Chem
from rdkit.Chem.QED import qed
from utils.sascorer import compute_sa_score
from utils.docking import QVinaDockingTask
from utils.scoring_func import get_logp, obey_lipinski

class RefineData(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # delete H atom of pocket
        protein_element = data.protein_element
        is_H_protein = (protein_element == 1)
        if torch.sum(is_H_protein) > 0:
            not_H_protein = ~is_H_protein
            data.protein_atom_name = list(compress(data.protein_atom_name, not_H_protein)) 
            data.protein_atom_to_aa_type = data.protein_atom_to_aa_type[not_H_protein]
            data.protein_element = data.protein_element[not_H_protein]
            data.protein_is_backbone = data.protein_is_backbone[not_H_protein]
            data.protein_pos = data.protein_pos[not_H_protein]
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = (ligand_element == 1)
        if torch.sum(is_H_ligand) > 0:
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            # nbh
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -np.ones(len(not_H_ligand), dtype=np.int64)
            index_changer[not_H_ligand] = np.arange(torch.sum(not_H_ligand))
            new_nbh_list = [value for ind_this, value in zip(not_H_ligand, data.ligand_nbh_list.values()) if ind_this]
            data.ligand_nbh_list = {i:[index_changer[node] for node in neigh if node not in index_atom_H] for i, neigh in enumerate(new_nbh_list)}
            # bond
            ind_bond_with_H = np.array([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in zip(*data.ligand_bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = torch.tensor(index_changer)[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]

        return data

class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True, 
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_num_bonds = torch.stack([
            self.count_neighbors(
                data.ligand_bond_index, 
                symmetry=True, 
                valence=(data.ligand_bond_type == i).long(),
                num_nodes=data.ligand_element.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        return data

class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1 + 1

    def __call__(self, data):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        is_mol_atom = torch.zeros_like(is_backbone, dtype=torch.long)
        # x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        x = torch.cat([element, amino_acid, is_backbone, is_mol_atom], dim=-1)
        data.protein_atom_feature = x
        # data.compose_index = torch.arange(len(element), dtype=torch.long)
        return data

class FeaturizeLigandAtom(object):
    
    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6,7,8,9,15,16,17])  # C N O F P S Cl
        assert len(self.atomic_numbers) == 7, NotImplementedError('fix the staticmethod: chagne_bond')
    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + (1 + 1 + 1) + 3 

    def __call__(self, data):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        # chem_feature = data.ligand_atom_feature
        is_mol_atom = torch.ones([len(element), 1], dtype=torch.long)
        n_neigh = data.ligand_num_neighbors.view(-1, 1)
        n_valence = data.ligand_atom_valence.view(-1, 1)
        ligand_atom_num_bonds = data.ligand_atom_num_bonds
        # x = torch.cat([element, chem_feature, ], dim=-1)
        x = torch.cat([element, is_mol_atom, n_neigh, n_valence, ligand_atom_num_bonds], dim=-1)
        data.ligand_atom_feature_full = x
        return data
    


class AtomComposer(object):

    def  __init__(self, protein_dim, ligand_dim, knn):
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.knn = knn  # knn of compose atoms
    
    def __call__(self, data):
        # fetch ligand context and protein from data
        ligand_context_pos = data.ligand_context_pos
        ligand_context_feature_full = data.ligand_context_feature_full
        protein_pos = data.protein_pos
        protein_atom_feature = data.protein_atom_feature
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)

        # compose ligand context and protein. save idx of them in compose
        data.compose_pos = torch.cat([ligand_context_pos, protein_pos], dim=0)
        len_compose = len_ligand_ctx + len_protein
        ligand_context_feature_full_expand = torch.cat([
            ligand_context_feature_full, torch.zeros([len_ligand_ctx, self.protein_dim - self.ligand_dim], dtype=torch.long)
        ], dim=1)
        data.compose_feature = torch.cat([ligand_context_feature_full_expand, protein_atom_feature], dim=0)
        data.idx_ligand_ctx_in_compose = torch.arange(len_ligand_ctx, dtype=torch.long)  # can be delete
        data.idx_protein_in_compose = torch.arange(len_protein, dtype=torch.long) + len_ligand_ctx  # can be delete

        # build knn graph and bond type
        data = self.get_knn_graph(data, self.knn, len_ligand_ctx, len_compose, num_workers=16)
        return data

    @staticmethod
    def get_knn_graph(data, knn, len_ligand_ctx, len_compose, num_workers=1, ):
        data.compose_knn_edge_index = knn_graph(data.compose_pos, knn, flow='target_to_source', num_workers=num_workers)

        id_compose_edge = data.compose_knn_edge_index[0, :len_ligand_ctx*knn] * len_compose + data.compose_knn_edge_index[1, :len_ligand_ctx*knn]
        id_ligand_ctx_edge = data.ligand_context_bond_index[0] * len_compose + data.ligand_context_bond_index[1]
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        data.compose_knn_edge_type = torch.zeros(len(data.compose_knn_edge_index[0]), dtype=torch.long)  # for encoder edge embedding
        data.compose_knn_edge_type[idx_edge[idx_edge>=0]] = data.ligand_context_bond_type[idx_edge>=0]
        data.compose_knn_edge_feature = torch.cat([
            torch.ones([len(data.compose_knn_edge_index[0]), 1], dtype=torch.long),
            torch.zeros([len(data.compose_knn_edge_index[0]), 3], dtype=torch.long),
        ], dim=-1) 
        data.compose_knn_edge_feature[idx_edge[idx_edge>=0]] = F.one_hot(data.ligand_context_bond_type[idx_edge>=0], num_classes=4)    # 0 (1,2,3)-onehot
        return data

def get_bfs_perm(nbh_list, start):
    num_nodes = len(nbh_list)#31
    num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])
    bfs_queue = [start]#initialize the bfs_queue [5]
    bfs_perm = []
    num_remains = [num_neighbors.clone()]#[tensor([3, 2, 2, 3, ... 3, 1, 1])]
    bfs_next_list = {}
    visited = {bfs_queue[0]}#{5}

    num_nbh_remain = num_neighbors.clone()#tensor([3, 2, 2, 3, ... 3, 1, 1])
    
    while len(bfs_queue) > 0:
        current = bfs_queue.pop(0)
        for nbh in nbh_list[current]:
            num_nbh_remain[nbh] -= 1
        bfs_perm.append(current)
        num_remains.append(num_nbh_remain.clone())

        next_candid = [] 
        for nxt in nbh_list[current]:
            if nxt in visited: continue
            next_candid.append(nxt)
            visited.add(nxt)
            
        random.shuffle(next_candid)
        bfs_queue += next_candid
        bfs_next_list[current] = copy.copy(bfs_queue)

    return torch.LongTensor(bfs_perm), bfs_next_list, num_remains

def get_data_list(data, bfs_perm):
    atomic_numbers = [6,7,8,9,15,16,17]
    
    target_list = []
    feat_list = []
    t=len(bfs_perm)
    for i in range(0,t):
        atom_idx = bfs_perm[i]
        element = data['ligand_element'][atom_idx].view(-1)
        element_idx = atomic_numbers.index(element)
        pos = data['ligand_pos'][atom_idx]
        target_list.append((torch.tensor([i]), torch.tensor([atom_idx]), torch.tensor([element_idx]), pos))
        feat_list.append(data['ligand_atom_feature_full'][i])
    
    actions = torch.stack([torch.cat(one) for one in target_list], dim=0)
    ligand_atom_feat = torch.stack(feat_list, dim=0)
    return actions, ligand_atom_feat




class MolDesign_dataset(Dataset):
    def __init__(self, 
                 root="/huyuqi/MolDesign/data/crossdocked_pocket10",split_path = '/huyuqi/MolDesign/data/split_by_name.pt',
                 mode = "train",
                 transform = None) :
                #  root="./data/crossdocked_pocket10", 
                #  split_path = './data/split_by_name.pt',
        super().__init__()
        dataset = PocketLigandPairDataset(root, transform, name = "full_prop")
        split_by_name = torch.load(split_path)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        if mode == "train":
            self.data = Subset(dataset, indices=split['train'])
        if mode == "test":
            self.data = Subset(dataset, indices=split['test'])
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            data = self.data[index]
            if data.ligand_pos.shape[0]>50: 
                return None
            
            metrics = {
                    "vina": data.ligand_vina,
                    "qed": data.ligand_qed,
                    "sa": data.ligand_sa,
                    "lipinski": data.ligand_lipinski,
                    "logp": data.ligand_logp}
            
            protein_mask = (data.protein_is_backbone&data.protein_is_N)|data.protein_is_in_5A
            return data.start_idx, data.protein_atom_feature[protein_mask], data.protein_pos[protein_mask], data.protein_filename, data.protein_one_knn_edge_index, data.ligand_smiles, metrics

        except:
            return None


