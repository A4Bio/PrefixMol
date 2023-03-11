import torch
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from torch.nn import Module, ModuleList, LeakyReLU, LayerNorm
from torch_scatter import scatter_sum, scatter_add, scatter_softmax
from math import pi as PI
from modules.common import GaussianSmearing, EdgeExpansion
from modules.invariant import GVLinear, VNLeakyReLU, MessageModule, GVPerceptronVN
from torch.nn import Module, Sequential
import torch.nn as nn
import math
GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)

class AtomEmbedding(Module):
    def __init__(self, in_scalar, in_vector,
                 out_scalar, out_vector, vector_normalizer=20.):
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)

    def forward(self, scalar_input, vector_input):
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (3, ), 'Not support. Only one vector can be input'
        sca_emb = self.emb_sca(scalar_input[:, :self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb

def embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein,
                                      ligand_atom_emb, protein_atom_emb,
                                      emb_dim):

    h_ligand = ligand_atom_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
    h_protein = protein_atom_emb(compose_feature[idx_protein], compose_pos[idx_protein])
    
    device = h_ligand[0].device
    h_sca = torch.zeros([len(compose_pos), emb_dim[0]],device=device)
    h_vec = torch.zeros([len(compose_pos), emb_dim[1], 3],device=device)
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]

class CFTransformerEncoderVN(Module):
    
    def __init__(self, hidden_channels=[256, 64], edge_channels=64, num_edge_types=4, key_channels=128, num_heads=4, num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels  # not use
        self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):#num_interactions:6
            block = AttentionInteractionBlockVN(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                num_edge_types=num_edge_types,
                key_channels=key_channels,
                num_heads=num_heads,
                cutoff = cutoff
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]
    
    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index, edge_feature):

        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]

        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h


class AttentionInteractionBlockVN(Module):

    def __init__(self, hidden_channels, edge_channels, num_edge_types, key_channels, num_heads=1, cutoff=10.):
        super().__init__()
        self.num_heads = num_heads
        # edge features
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels - num_edge_types)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        ## compare encoder and classifier message passing

        # edge weigths and linear for values
        self.message_module = MessageModule(hidden_channels[0], hidden_channels[1], edge_channels, edge_channels,
                                                                                hidden_channels[0], hidden_channels[1], cutoff)

        # centroid nodes and finall linear
        self.centroid_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1])
        self.out_transform = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, x, edge_index, edge_feature, edge_vector):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index   # (E,) , (E,)

        # Compute edge features
        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector) 

        msg_j_sca, msg_j_vec = self.message_module(x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=True)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)  #.view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)  #.view(N, -1, 3) # (N, heads*H_per_head, 3)
        x_out_sca, x_out_vec = self.centroid_lin(x)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


class PositionPredictor(Module):
    def __init__(self, in_sca, in_vec, num_filters, n_component):
        super().__init__()
        self.n_component = n_component
        self.gvp = Sequential(
            GVPerceptronVN(in_sca, in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.mu_net = GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.logsigma_net= GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.pi_net = GVLinear(num_filters[0], num_filters[1], n_component, 1)

    def forward(self, h_focal, pos_focal):
        # h_focal = [h[idx_focal] for h in h_compose] # [8,256] [8,64,3]
        # pos_focal = pos_compose[idx_focal] # [8,3]
        
        feat_focal = self.gvp(h_focal)
        relative_mu = self.mu_net(feat_focal)[1]  # (N_focal, n_component, 3)
        logsigma = self.logsigma_net(feat_focal)[1]  # (N_focal, n_component, 3)
        sigma = torch.exp(logsigma)
        pi = self.pi_net(feat_focal)[0]  # (N_focal, n_component)
        pi = F.softmax(pi, dim=1)
        
        abs_mu = relative_mu + pos_focal.unsqueeze(dim=1).expand_as(relative_mu)
        return relative_mu, abs_mu, sigma, pi

    def get_mdn_probability(self, mu, sigma, pi, pos_target):
        prob_gauss = self._get_gaussian_probability(mu, sigma, pos_target)
        prob_mdn = pi * prob_gauss
        prob_mdn = torch.sum(prob_mdn, dim=1)
        return prob_mdn
    
    def _get_gaussian_probability(self, mu, sigma, pos_target):
        """
        mu - (N, n_component, 3)
        sigma - (N, n_component, 3)
        pos_target - (N, 3)
        """
        target = pos_target.unsqueeze(1).expand_as(mu)
        errors = target - mu 
        sigma = sigma + 1e-16
        p = GAUSSIAN_COEF * torch.exp(- 0.5 * (errors / sigma)**2) / sigma
        p = torch.prod(p, dim=2)
        return p # (N, n_component)

    def sample_batch(self, mu, sigma, pi, num):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, 3)
            sigma - (N_batch, n_cat, 3)
            pi - (N_batch, n_cat)
        return
            (N_batch, num, 3)
        """
        index_cats = torch.multinomial(pi, num, replacement=True)  # (N_batch, num)
        # index_cats = index_cats.unsqueeze(-1)
        index_batch = torch.arange(len(mu)).unsqueeze(-1).expand(-1, num)  # (N_batch, num)
        mu_sample = mu[index_batch, index_cats]  # (N_batch, num, 3)
        sigma_sample = sigma[index_batch, index_cats]
        values = torch.normal(mu_sample, sigma_sample)  # (N_batch, num, 3)
        return values

    def get_maximum(self, mu, sigma, pi):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, 3)
            sigma - (N_batch, n_cat, 3)
            pi - (N_batch, n_cat)
        return
            (N_batch, n_cat, 3)
        """
        return mu


class SpatialClassifierVN(Module):

    def __init__(self, num_classes, in_sca, in_vec, num_filters, edge_channels, num_heads, k=32, cutoff=10.0, num_bond_types=3):
        super().__init__()
        self.num_bond_types = num_bond_types
        self.message_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff)

        self.nn_edge_ij = Sequential(
            GVPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        
        self.classifier = Sequential(
            GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_classes, 1)
        )


        
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)

        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos_query, pos_compose, node_attr_compose, edge_index_q_cps_knn):
        # (self, pos_query, edge_index_query, pos_ctx, node_attr_ctx, is_mol_atom, batch_query, batch_edge, batch_ctx):
        """
        Args:
            pos_query:   (N_query, 3)
            edge_index_query: (2, N_q_c, )
            pos_ctx:     (N_ctx, 3)
            node_attr_ctx:  (N_ctx, H)
            is_mol_atom: (N_ctx, )
            batch_query: (N_query, )
            batch_ctx:   (N_ctx, )
        Returns
            (N_query, num_classes)
        """

        # Pairwise distances and contextual node features
        vec_ij = pos_query[edge_index_q_cps_knn[0]] - pos_compose[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        
        # node_attr_ctx_j = [node_attr_ctx_[edge_index_q_cps_knn[1]] for node_attr_ctx_ in node_attr_ctx]  # (A, H)
        h = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True)

        # Aggregate messages
        y = [scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)), # (N_query, F)
                scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0))]

        # element prediction
        y_cls, _ = self.classifier(y)  # (N_query, num_classes)

        return y_cls

class FrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.time_embed = nn.Embedding(100,in_sca)
        self.att_net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, 1, 1)
        )
        self.net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, in_sca, in_vec)
        )

    def forward(self, h_att, pos_context, t, batch_id):
        N, _ = h_att[0].shape
        t = torch.tensor([t]*N, device=pos_context.device).long()
        time_embedding = self.time_embed(t)
        h_att = [h_att[0]+time_embedding, h_att[1]]
        att = self.att_net(h_att)[0].view(-1,1)
        h_att = self.net(h_att)
        att = scatter_softmax(att, batch_id, dim=0)
        pos_focal = scatter_sum(att*pos_context, batch_id, dim=0)
        feat_focal = scatter_sum(att*h_att[0], batch_id, dim=0)
        vec_focal = scatter_sum(att.view(-1,1,1)*h_att[1], batch_id, dim=0)
        return [feat_focal, vec_focal], pos_focal