import torch
from torch.nn import Module
import torch.nn as nn
from torch.nn import functional as F
from utils.misc import unique
from modules.MolDesign_module import embed_compose, AtomEmbedding, CFTransformerEncoderVN, PositionPredictor, SpatialClassifierVN, FrontierLayerVN
from torch_geometric.nn.pool import knn_graph, radius, knn
from torch_scatter import scatter_sum
import time
import os
from transformers import GPT2Config
from modules.modeling_gpt2 import GPT2LMHeadModel
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from rdkit import Chem
from rdkit.Chem import Draw
import re
import partialsmiles as ps
import numpy as np

atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
ATOM_VALENCY = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence

def correct_mol(x):
    xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])
              
    return mol



class Moldesign_model(Module):

    def __init__(self, 
                 hidden_channels, 
                 edge_channels, 
                 key_channels, 
                 num_heads, 
                 num_interactions, 
                 knn_enc,
                 knn_field,
                 cutoff, 
                 hidden_channels_vec, 
                 num_filters, 
                 num_filters_vec, 
                 pos_n_component, 
                 protein_atom_feature_dim, 
                 ligand_atom_feature_dim, 
                 num_classes):
        super().__init__()
        self.knn_enc = knn_enc
        self.knn_field = knn_field
        self.emb_dim = [hidden_channels, hidden_channels_vec]
        self.protein_atom_emb = AtomEmbedding(protein_atom_feature_dim, 1, *self.emb_dim)
        self.ligand_atom_emb = AtomEmbedding(ligand_atom_feature_dim, 1, *self.emb_dim)
        self.encoder3d = CFTransformerEncoderVN(
                            hidden_channels = [hidden_channels, hidden_channels_vec],
                            edge_channels = edge_channels,
                            key_channels = key_channels,  # not use
                            num_heads = num_heads,  # not use
                            num_interactions = num_interactions,
                            k = knn_enc,
                            cutoff = cutoff,
                        )
        
        self.tokenizer = SmilesTokenizer("/huyuqi/MolDesign/data/vocab.txt")

        config = GPT2Config.from_dict({'_name_or_path': 'gpt2', 'activation_function': 'gelu_new', 'architectures': ['GPT2LMHeadModel'], 'attn_pdrop': 0.1, 'bos_token_id': 590, 'embd_pdrop': 0.1, 'eos_token_id': 590, 'initializer_range': 0.02, 'layer_norm_epsilon': 1e-05, 'model_type': 'gpt2', 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_inner': None, 'n_layer': 12, 'n_positions': 1024, 'reorder_and_upcast_attn': False, 'resid_pdrop': 0.1, 'scale_attn_by_inverse_layer_idx': False, 'scale_attn_weights': True, 'summary_activation': None, 'summary_first_dropout': 0.1, 'summary_proj_to_labels': True, 'summary_type': 'cls_index', 'summary_use_proj': True, 'task_specific_params': {'text-generation': {'do_sample': True, 'max_length': 160, 'min_length': 10, 'prefix': '<|endoftext|>', 'temperature': 1.0, 'top_p': 0.95}}, 'torch_dtype': 'float32', 'transformers_version': '4.25.1', 'use_cache': True, 'vocab_size': self.tokenizer.vocab_size, '_commit_hash': '2a9b3a159b6e9ae4a9722e2f603d40f0b50b8117', "condition_dim":hidden_channels})
        
        self.pocket_embed = nn.Linear(hidden_channels, 768)
        self.vina_embed = nn.Linear(1, 768)
        self.qed_embed = nn.Linear(1, 768)
        self.logp_embed = nn.Linear(1, 768)
        self.sa_embed = nn.Linear(1, 768)
        self.lipinski_embed = nn.Linear(1, 768)
        self.decoderseq = GPT2LMHeadModel(config)
        

    
    def compose_feat(self, protein_feature, protein_pos, ligand_feature, ligand_pos, protein_batch_id, ligand_batch_id):
        max_feat = max(protein_feature.shape[1], ligand_feature.shape[1]) 
        PAD = lambda x: F.pad(x, (0, max_feat-x.shape[-1]))
        compose_feature = torch.cat([protein_feature, PAD(ligand_feature)], dim=0)#[3564, 27]
        compose_pos = torch.cat([protein_pos, ligand_pos], dim=0)#[3564, 3]
        compose_batch_id = torch.cat([protein_batch_id, ligand_batch_id], dim=0)#[3564]
        idx = torch.arange(protein_feature.shape[0]+ligand_feature.shape[0], device=protein_feature.device)#[3564]
        idx_protein = idx[:protein_feature.shape[0]]
        idx_ligand = idx[protein_feature.shape[0]:]
        return compose_feature, idx_protein, idx_ligand, compose_batch_id, compose_pos
    


    
    def forward(self, batch, smiles_list, metrics_list, protein_file_list, mode='train', opt_config=None, get_attn_map=False):

        start_idx, protein_feature, protein_pos,  protein_batch_id, protein_edge_idx = batch 
        device = protein_feature.device
        N_atoms_protein = scatter_sum(torch.ones_like(protein_batch_id), protein_batch_id)
        shift_protein = torch.cumsum(torch.cat([torch.zeros([1], device=device).long(), N_atoms_protein]), dim=-1)[:-1]
        
        
       
        idx_context = start_idx+shift_protein
                                
        h_compose = self.protein_atom_emb(protein_feature, protein_pos)
        compose_pos = protein_pos
        compose_knn_edge_index = protein_edge_idx
        
        N_edge = compose_knn_edge_index.shape[1]
        compose_knn_edge_feature = torch.zeros(N_edge,4, device = device)
        
        h_compose = self.encoder3d(
                    node_attr = h_compose,
                    pos = compose_pos,
                    edge_index = compose_knn_edge_index,
                    edge_feature = compose_knn_edge_feature,
                )
        
        feat3d = h_compose[0][idx_context] #[99,256]

        # len(metrics_list)=89
        if opt_config is None:
            vina = torch.tensor([one['vina'] for one in metrics_list], device=device).reshape(-1,1).float() #[99,1]
            qed = torch.tensor([one['qed'] for one in metrics_list], device=device).reshape(-1,1).float() #[99,1]
            logp = torch.tensor([one['logp'] for one in metrics_list], device=device).reshape(-1,1).float() #[99,1]
            sa = torch.tensor([one['sa'] for one in metrics_list], device=device).reshape(-1,1).float() #[99,1]
            lipinski = torch.tensor([one['lipinski'] for one in metrics_list], device=device).reshape(-1,1).float()#[99,1]
        else:
            vina = torch.tensor([one['vina'] for one in metrics_list], device=device).reshape(-1,1).float() + opt_config["vina"] #[99,1]
            qed = torch.tensor([one['qed'] for one in metrics_list], device=device).reshape(-1,1).float() + opt_config["qed"]#[99,1]
            logp = torch.tensor([one['logp'] for one in metrics_list], device=device).reshape(-1,1).float()+ opt_config["logp"] #[99,1]
            sa = torch.tensor([one['sa'] for one in metrics_list], device=device).reshape(-1,1).float()+ opt_config["sa"] #[99,1]
            lipinski = torch.tensor([one['lipinski'] for one in metrics_list], device=device).reshape(-1,1).float()+ opt_config["lipinski"]#[99,1]
        
        feat3d = torch.stack([self.pocket_embed(feat3d),self.vina_embed(vina), self.qed_embed(qed), self.logp_embed(logp), self.sa_embed(sa), self.lipinski_embed(lipinski)], dim=1)#[99, 6, 768]
        
        feature_dict = self.tokenizer(smiles_list, return_tensors='pt', add_special_tokens=False, padding=True).to(device) #[99, 5, 768]
        del feature_dict['token_type_ids']
 
        if mode == "train":
            x = feature_dict['input_ids'][:,:-1] 
            y = feature_dict['input_ids'][:,1:]
            feature_dict['input_ids'] = x
            y_mask = feature_dict['attention_mask'][:,1:]
            feature_dict['attention_mask'] = feature_dict['attention_mask'][:,:-1]
            feature_dict['feat_3d'] = feat3d
            feature_dict['mode'] = "train"
            # add feat_prop
            
            prediction = self.decoderseq(**feature_dict, output_hidden_states=True)
            # logits = prediction.logits
            
            # ------------insert condition--------------
            condition_length = feat3d.shape[1]
            logits = prediction.logits[:,condition_length:,:]
            # ------------------------------------------
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction ='none')
            loss = (loss*y_mask.reshape(-1)).mean()
            
            return loss
        
        if mode=='test':
            feature_dict['feat_3d'] = feat3d
            feature_dict['input_ids'] = feature_dict['input_ids'][:,:1]
            feature_dict['attention_mask'] = feature_dict['attention_mask'][:,:1] 
            feature_dict['current_smiles'] = self.tokenizer.batch_decode(feature_dict['input_ids'])
            feature_dict['tokenizer'] = self.tokenizer
            feature_dict['mode'] = "test"

                
            prediction = self.decoderseq.generate(**feature_dict, 
                                                    max_length=50, 
                                                    num_beams=5, 
                                                    early_stopping=True)
            
            if get_attn_map:
                attention_map = self.decoderseq.attentions[0][0,:,:6,:6].cpu().numpy()
                with open('/huyuqi/MolDesign/results/attn_map/vina{}_qed{}_sa{}_lip{}_logp{}.npy'.format(*list(opt_config.values())), 'wb') as f:
                    np.save(f,attention_map)
                return None
            
            smiles_list = []
            valid = 0
            invalid = 0
            protein_file_list.reverse()
            for i in range(prediction.shape[0]):
                smiles = self.tokenizer.decode(prediction[i], skip_special_tokens=True).replace(" ","")
                # add the corresponding pdb file
                smiles_list.append(protein_file_list.pop())
                for t in range(len(smiles)-1,-1,-1):
                    try: 
                        ps.ParseSmiles(smiles[:t], partial=True)
                        mol = Chem.MolFromSmiles(smiles[:t])
                        Draw.MolToFile(mol, f"/huyuqi/MolDesign/generated/test_{i}.png")
                        valid += 1
                        smiles_list.append(smiles[:t])
                        # print(valid)
                        break
                    except:
                        invalid += 1
                        pass

            with open("/huyuqi/MolDesign/generated/smiles.txt", "w") as f:
                f.writelines("\n".join(smiles_list))

            print("the number of invalid molecules: {}".format(invalid))
            return smiles_list
            