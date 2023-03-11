import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from torch.utils.data import Dataset
import copy
import os
import sys
sys.path.append('.')
import uuid
from itertools import compress

import torch.nn.functional as F
from torch_geometric.nn.pool import knn_graph
from torch_geometric.transforms import Compose
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.nn import knn, radius
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

try:
    from .data import ProteinLigandData
    from .datasets import *
    from .misc import *
    from .protein_ligand import ATOM_FAMILIES
except:
    from utils.data import ProteinLigandData
    from utils.datasets import *
    from utils.misc import *
    from utils.protein_ligand import ATOM_FAMILIES
import argparse
import logging

PeriodicTable = ["", "H", "He", "Li", "Be", "B",
                 "C", "N", "O", "F", "Ne",
                 "Na", "Mg", "Al", "Si", "P",
                 "S", "Cl", "Ar", "K", "Ca",
                 "Sc", "Ti", "V", "Cr", "Mn",
                 "Fe", "Co", "Ni", "Cu", "Zn"]

AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

AA_NAME_NUMBER = {
    k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
}

AA_NUMBER_NAME = {
    i: k for i, (k, _) in enumerate(AA_NAME_SYM.items())
}

BACKBONE_NAMES = ["CA", "C", "N", "O"]

def pdb_decode_protein_ele(protein_ele):

    return protein_ele

def pdb_decode_protein_atom_to_aa_type(protein_atom_to_aa_type):
    return AA_NUMBER_NAME[int(protein_atom_to_aa_type)]

def pdb_decode_ligand_ele(ligand_ele):

    return ligand_ele


def get_bfs_perm(nbh_list):
    num_nodes = len(nbh_list)#31
    num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])

    bfs_queue = [random.randint(0, num_nodes-1)]#initialize the bfs_queue [5]
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


def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)


def get_data_list(data, bfs_perm):
    data_list = []
    t=len(bfs_perm)
    for i in range(t):
        data_list.append((i,data['ligand_element'][bfs_perm[:i]],data['ligand_pos'][bfs_perm[:i]]))
    return data_list


def format_str(index, name, res, chain_id, res_seq, x, y, z, occu, temp, elem, mode="ATOM"):
    mode = mode.ljust(6, " ")
    index = str(index).rjust(5, " ")
    name = name.ljust(4, " ")
    res_seq = str(res_seq).rjust(4, " ")
    x = str(round(float(x), 3)).rjust(8, " ") if x != "" else ""
    y = str(round(float(y), 3)).rjust(8, " ") if y != "" else ""
    z = str(round(float(z), 3)).rjust(8, " ") if z != "" else ""
    occu = ("%.2f" % float(occu)).rjust(6, " ") if occu != "" else ""
    temp = str(temp).rjust(6, " ") if temp != "" else ""
    elem = elem.rjust(2, " ") if elem != "" else ""
    formatted_str = f"{mode}{index}  {name}{res} {chain_id}{res_seq}    {x}{y}{z}{occu}{temp}          {elem}  "
    return formatted_str

if __name__ == '__main__': 
    from munch import DefaultMunch
    config = DefaultMunch.fromDict({'name': 'pl', 'path': './data/crossdocked_pocket10', 'split': './data/split_by_name.pt'})
    dataset, subsets = get_dataset(
        config = config,
    )

    data = subsets['test'][0]
    #first show the original protein and ligand
    #protein——ATOM
    protein_ele = data['protein_element'].tolist()
    protein_pos = data['protein_pos'].tolist()
    protein_atom_to_aa_type = data['protein_atom_to_aa_type'].tolist()
    
    atom_length = data['protein_element'].shape[0]
    
    text_list = []
    idx = 1
    for i in range(atom_length):
        atom_name = data['protein_atom_name'][i]
        res = pdb_decode_protein_atom_to_aa_type(data['protein_atom_to_aa_type'][i])
        chain_id = "A" if data['protein_is_backbone'][i] else "B"
        res_seq = ""
        x, y, z = float(data['protein_pos'][i][0]), float(data['protein_pos'][i][1]), float(data['protein_pos'][i][2])
        elem = PeriodicTable[data['protein_element'][i]]
        occu = 1.00
        temp = 55.23
        text_list.append(format_str(idx, atom_name, res, chain_id, res_seq, x, y, z, occu, temp, elem))
        idx += 1   
    text_list.append(format_str(idx, atom_name, res, chain_id, res_seq, "", '', '', '', '', '', "TER"))
    
    #bfs traversal was performed on ligand and the traversal order was recorded
    bfs_perm,bfs_next_list,num_remains = get_bfs_perm(data['ligand_nbh_list'])

    # #every state (t∈[0,n]) of ligand
    ligand_state_list = get_data_list(data, bfs_perm.tolist())
    

    for i, ligand_state in enumerate(ligand_state_list):
        tmp_text = copy.deepcopy(text_list)
        
        ligand_ele = ligand_state[1].tolist()
        ligand_pos = ligand_state[2].tolist()
   
        print(len(tmp_text))
        hetatm_lenght = len(ligand_ele)
        print(hetatm_lenght)
        idx += 1
        for i in range(hetatm_lenght):
            atom_name = "  "
            res = "  "
            chain_id = "  "
            res_seq = "  "
            x, y, z = float(ligand_pos[i][0]), float(ligand_pos[i][1]), float(ligand_pos[i][2])
            elem = PeriodicTable[ligand_ele[i]]
            occu = 1.00
            temp = 55.23
            tmp_text.append(format_str(idx, atom_name, res, chain_id, res_seq, x, y, z, occu, temp, elem, "HETATM"))
            idx += 1 
        tmp_text.append("END")
        print(len(tmp_text))
        os.makedirs("pdb_data", exist_ok=True)
        with open(f"pdb_data/data_{i}.pdb", "w", encoding='gbk') as f:
            f.writelines("\n".join(tmp_text))
        f.close()
  