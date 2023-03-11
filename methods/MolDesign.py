import os
import utils
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from tqdm import tqdm
from typing import *
from methods.base_method import Base_method
from torch_scatter import scatter_sum, scatter_mean, scatter_max
from models.MolDesign_model import Moldesign_model
from utils.visual_pred_pos import visual_pred_pos
import time
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d import _C
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from utils.mol_utils import reconstruct_from_generated, get_metric
from rdkit import Chem

from rdkit.Geometry import Point3D

def get_rdkit_coords(mol):
    from rdkit.Chem import AllChem
    # mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)

    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return torch.tensor(lig_coords, dtype=torch.float32)


class MolDesign(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

    def _build_model(self):
        model = Moldesign_model(
                 hidden_channels = self.args.hidden_channels, 
                 edge_channels = self.args.edge_channels, 
                 key_channels = self.args.key_channels, 
                 num_heads = self.args.num_heads, 
                 num_interactions = self.args.num_interactions, 
                 knn_enc = self.args.knn_enc,
                 knn_field = self.args.knn_field,
                 cutoff = self.args.cutoff, 
                 hidden_channels_vec = self.args.hidden_channels_vec, 
                 num_filters = self.args.num_filters, 
                 num_filters_vec = self.args.num_filters_vec, 
                 pos_n_component = self.args.pos_n_component, 
                 protein_atom_feature_dim = self.args.protein_atom_feature_dim, 
                 ligand_atom_feature_dim = self.args.ligand_atom_feature_dim, 
                 num_classes = self.args.num_classes+1
                  ).to(self.device)
        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)
        model.to(self.device)
        return model
    
   
    def get_loss(self, results):
       
        pred_atom_list = results["pred_atom"]
        pred_pos_list = results["pred_pos"]
        true_atom = results["true_atom"]
        true_pos = results["true_pos"]
        label_mask = results["label_mask"]
        
        loss_report=[]
        atom_loss_report=[]
        pos_loss_report=[]
        for idx, pred_pos in enumerate(pred_pos_list):
           
            pos_loss, index, _ = chamfer_distance(pred_pos, true_pos) 

            index = index.squeeze(-1)
            assert true_atom.shape == index.shape
            true_atom = torch.gather(true_atom, dim=1, index= index)
        
            atom_loss = self.criterion(pred_atom_list[idx].permute(0,2,1), true_atom.long())
            atom_loss = (atom_loss*label_mask).sum(dim=1)/label_mask.sum(dim=1)
            atom_loss = atom_loss.mean()

            loss = atom_loss + pos_loss

            atom_loss_report.append(atom_loss)
            pos_loss_report.append(pos_loss)
            loss_report.append(loss)

        loss=sum(loss_report)
        atom_loss=sum(atom_loss_report)
        pos_loss=sum(pos_loss_report)

        return loss, atom_loss.detach().cpu().numpy(), pos_loss.detach().cpu().numpy()
        
        
    
    def train_one_epoch(self, train_loader): 
        self.model.train()
       
        self.train_loader = train_loader
        train_pbar = tqdm(train_loader)
       
        loss_list = []
        
        for idx, batch in enumerate(train_pbar):
           
            protein_file_list = batch[-3]
            smiles_list = batch[-2]
            metrics_list = batch[-1]
            
            batch = [one.to(self.device) for one in batch[:-3]]
           
            self.optimizer.zero_grad()

            loss = self.model(batch, smiles_list, metrics_list, protein_file_list)
            loss_list.append(loss.item())

            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            
        epoch_metric = {"loss": np.mean(loss_list)}
        self.scheduler.step()
        
        return epoch_metric

    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        valid_pbar = tqdm(valid_loader)

        loss_list = []
        for batch in valid_pbar:
            protein_file_list = batch[-3]
            smiles_list = batch[-2]
            metrics_list = batch[-1]
            
            batch = [one.to(self.device) for one in batch[:-3]]
            with torch.no_grad():
                loss = self.model(batch, smiles_list, metrics_list, protein_file_list)
           
            loss_list.append(loss.item())
            
    
        epoch_metric = {"loss": np.mean(loss_list)}
        return epoch_metric

    def test_one_epoch(self, test_loader, opt_config=None):
        self.model.eval()
        test_pbar = tqdm(test_loader)
        loss_list = []
        
        metrics = []
        for batch in test_loader:
            protein_file_list = batch[-3]
            smiles_list = batch[-2]
            metrics_list = batch[-1]
            
            batch = [one.to(self.device) for one in batch[:-3]]
            
            start_idx, protein_feature, protein_pos,  protein_batch_id, protein_edge_idx = batch 
            device = protein_pos.device
            N_atoms_protein = scatter_sum(torch.ones_like(protein_batch_id), protein_batch_id)
            shift_protein = torch.cumsum(torch.cat([torch.zeros([1], device=device).long(), N_atoms_protein]), dim=-1)[:-1]
            
            idx_context = start_idx+shift_protein 

            with torch.no_grad():
                        
                pred_smiles = self.model(batch, smiles_list, metrics_list, protein_file_list, mode='test', opt_config=opt_config)
                # clean the pred_smiles to split the pdb file
                WordFilter = set(['pdb'])
                pred_smiles_list = [smiles for smiles in pred_smiles if not any(word in smiles for word in WordFilter)]
                gt_vina = [i['vina'] for i in metrics_list]
                for b in range(len(pred_smiles_list)):
                    try:
                        mol = Chem.MolFromSmiles(pred_smiles_list[b])
                        rd_coords = get_rdkit_coords(mol) 
                        
                        center = protein_pos[idx_context][b].cpu().tolist()
                        metric = get_metric(protein_file_list[b], mol, center = center)
                        metrics.append(list((metric,gt_vina[b]))) 
                        
                    except:
                        pass


        # add mean and variance
        tmp_metric_list = [one[0] for one in metrics]
        gt_list = [one[1] for one in metrics] #vina ground truth
        # vina
        tmp = [one['vina'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10]
        vina_mean = np.mean([one for one in tmp if one is not None])
        vina_std = np.std([one for one in tmp if one is not None])
        #qed
        qed_mean = np.mean([one['qed'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        qed_std = np.std([one['qed'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        #sa
        sa_mean = np.mean([one['sa'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        sa_std = np.std([one['sa'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        #logp
        logp_mean = np.mean([one['logp'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        logp_std = np.std([one['logp'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        #lipinski
        lipinski_mean = np.mean([one['lipinski'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        lipinski_std = np.std([one['lipinski'] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])        
        
        #High Affinity
        count = 0
        tmp_vina = [one['vina'] for one in tmp_metric_list if one['rdmol']]
        for pred_vina, gt in zip(tmp_vina,gt_list):
            if pred_vina is not None and pred_vina <= gt:
                count = count + 1
        high_aff = count /len(metrics)
        
        #Diversity
        diversity_mean = np.mean([one['Tanimoto similarity'][1][0] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        diversity_std = np.std([one['Tanimoto similarity'][1][0] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])  
        #Sim_Train
        sim_train_mean = np.mean([one['Tanimoto similarity'][0] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])
        sim_train_std = np.std([one['Tanimoto similarity'][0] for one in tmp_metric_list if one['rdmol'].GetNumAtoms()>10])  

        epoch_metric = {"vina_mean": vina_mean, "vina_std": vina_std,
            "qed_mean": qed_mean, "qed_std": qed_std,
            "sa_mean": sa_mean, "sa_std": sa_std,
            "logp_mean":logp_mean, "logp_std":logp_std,
            "lipinski_mean": lipinski_mean, "lipinski_std": lipinski_std,
            "high_aff":high_aff,
            "diversity_mean": diversity_mean, "diversity_std": diversity_std,
            "sim_train_mean": sim_train_mean, "sim_train_std": sim_train_std
            }
        
        
        return epoch_metric

    def save_attn_map(self, test_loader, opt_config=None):
        self.model.eval()
        test_pbar = tqdm(test_loader)

        loss_list = []
        
        metrics = []
        for batch in test_loader:
            protein_file_list = batch[-3]
            smiles_list = batch[-2]
            metrics_list = batch[-1]
            
            batch = [one.to(self.device) for one in batch[:-3]]
            
            start_idx, protein_feature, protein_pos,  protein_batch_id, protein_edge_idx = batch 
            device = protein_pos.device
            N_atoms_protein = scatter_sum(torch.ones_like(protein_batch_id), protein_batch_id)
            shift_protein = torch.cumsum(torch.cat([torch.zeros([1], device=device).long(), N_atoms_protein]), dim=-1)[:-1]
            
            idx_context = start_idx+shift_protein

            with torch.no_grad():
                self.model(batch, smiles_list, metrics_list, protein_file_list, mode='test', opt_config=opt_config, get_attn_map=True)
            break
        return None
                
                

    def chamfer_distance(self, x, y, label_mask):
        d = torch.pow(x[:,:,None] - y[:,None], 2).sum(dim=-1)
        mask = label_mask[:,:,None]&label_mask[:,None]
        d = d*mask +1e9*(~mask)
        
        min_j_idx = d.min(dim=-1)[1].unsqueeze(-1).expand_as(d)
        min_i_idx = d.min(dim=-2)[1].unsqueeze(-2).expand_as(d)
        
        min_for_each_x_i = torch.gather(d, dim=-1, index = min_j_idx)[:,:,0]
        min_for_each_y_j = torch.gather(d, dim=-2, index = min_i_idx)[:,0,:]
    
        distance = min_for_each_x_i + min_for_each_y_j
      

        distance = (distance*label_mask).sum(dim=-1) 
        
        distance = torch.mean(distance)

        return distance, min_j_idx, min_i_idx

    
    