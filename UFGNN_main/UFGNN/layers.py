
import numpy as np
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, to_undirected
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
np.set_printoptions(threshold=np.inf)
from training.tools import *
import math

class MSTDM(nn.Module):
    def __init__(self, input_dim, scale_num, dropout, rnn_unit):
        """
        Multi-Scale Temporal Decomposition Module (MSTDM).

        Args:
            input_dim (int): Input feature dimension (C), e.g., 5.
            scale_num (int): Number of scales (M+1), minimum 1 (original scale only).
            dropout (float): Dropout rate for GRU.
            rnn_unit (int): Number of hidden units in GRU (F).
        """
        super(MSTDM, self).__init__()

        self.input_dim = input_dim  # C, feature dimension
        self.scale_num = max(1, scale_num)  # M+1, number of scales
        self.dropout = dropout
        self.rnn_unit = rnn_unit  # F, GRU hidden units

        # GRU layers for each scale
        self.grus = nn.ModuleList([
            nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.rnn_unit,
                num_layers=1,
                batch_first=True,
                dropout=self.dropout if self.dropout > 0 else 0
            ) for _ in range(self.scale_num)
        ])

    def forward(self, x):
        """
        Forward pass of MSTDM.

        Args:
            x (torch.Tensor): Input tensor of shape [T, N, C], where
                T: time window length,
                N: number of stocks,
                C: feature dimension.

        Returns:
            torch.Tensor: Multi-scale features of shape [N, (M+1), F].
        """
        T, N, C = x.shape  # [T, N, C]
        assert C == self.input_dim, f"Expected input_dim {self.input_dim}, got {C}"

        # List to store features for each scale
        multi_scale_features = []

        # Process each scale
        for m in range(self.scale_num):
            # Determine pooling window size
            pool_size = 1 if m == 0 else 2 * m

            # Average pooling for scale m
            # Reshape to [N, C, T] for AvgPool1d, then back to [T', N, C]
            x_m = x.permute(1, 2, 0)  # from [T, N, C] to [N, C, T]
            pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
            x_m = pool(x_m)  # [N, C, T']
            x_m = x_m.permute(2, 0, 1)  # [T', N, C], where T' = floor(T / pool_size)

            # Process with GRU
            # GRU expects [N, T', C], so reshape from [T', N, C] to [N, T', C]
            x_m = x_m.permute(1, 0, 2)  # [N, T', C]
            gru = self.grus[m]
            gru_out, _ = gru(x_m)  # [N, T', F]

            # Take the last time step's hidden state
            h_m = gru_out[:, -1, :]  # [N, F]
            multi_scale_features.append(h_m)

        # Concatenate features across all scales
        # Stack along the scale dimension: [N, (M+1), F]
        multi_scale_features = torch.stack(multi_scale_features, dim=1)  # [N, (M+1), F]

        return multi_scale_features


class TMIRIM(nn.Module):
    def __init__(self, input_dim, scale_num, n_hid, beta=1.0):
        """
        Tensor-based Multi-Scale Implicit Relational Inference Module (TMIRIM).

        Args:
            input_dim (int): Input feature dimension (F), e.g., 64 (from MSTDM rnn_unit).
            scale_num (int): Number of scales (M+1), e.g., 3.
            n_hid (int): Hidden dimension for internal feature mapping.
            beta (float): Hyperparameter for threshold filtering (default: 1.0).
        """
        super(TMIRIM, self).__init__()

        self.input_dim = input_dim  # F, input feature dimension
        self.scale_num = scale_num  # M+1, number of scales
        self.n_hid = n_hid  # Hidden dimension
        self.beta = beta  # Hyperparameter for threshold

        # Third-order parameter tensor T: [M+1, M+1, F]
        self.T = nn.Parameter(torch.randn(scale_num, scale_num, n_hid))

        # Global parameter vector alpha: [F]
        self.alpha = nn.Parameter(torch.randn(n_hid))

        # Linear layer to project input features to n_hid dimension
        self.proj = nn.Linear(input_dim, n_hid)

    def forward(self, h_tilde):
        """
        Forward pass of TMIRIM.

        Args:
            h_tilde (torch.Tensor): Multi-scale features from MSTDM, shape [N, (M+1), F].

        Returns:
            torch.Tensor: Relationship tensor R^t, shape [N, N, (M+1)].
        """
        N, M_plus_1, Fea = h_tilde.shape  # [N, (M+1), F]
        assert M_plus_1 == self.scale_num, f"Expected scale_num {self.scale_num}, got {M_plus_1}"
        assert Fea == self.input_dim, f"Expected input_dim {self.input_dim}, got {Fea}"

        # Project features to n_hid dimension
        h_tilde = self.proj(h_tilde)  # [N, (M+1), n_hid]

        # Initialize directional correlation tensor r: [N, N, (M+1)]
        r = torch.zeros(N, N, self.scale_num, device=h_tilde.device)

        # Compute directional correlations r_{i->j,k}^t for each k
        # Note: k in formula is from 1 to M+1, but in code we use 0 to M
        for k in range(self.scale_num):  # k from 0 to M, corresponds to 1 to M+1 in formula
            # Extract h_{i,k}^t for all i: [N, n_hid]
            h_i_k = h_tilde[:, k, :]  # [N, n_hid]

            # Compute correlation vector h_j^t * h_{i,k}^t: [N, N, (M+1)]
            # h_j^t: [N, (M+1), n_hid], h_i_k: [N, n_hid]
            h_j = h_tilde  # [N, (M+1), n_hid]
            # h_j[j,m,d] * h_i_k[i,d] -> corr[i,j,m]
            corr = torch.einsum('jmd,id->ijm', h_j, h_i_k)  # [N, (M+1), n_hid] [N, n_hid] = [N, N, (M+1)]

            # Apply T[k, :, :]: [M+1, n_hid]
            T_k = self.T[k, :, :]  # [M+1, n_hid]
            # Compute T_k * corr: [N, N, (M+1)] * [M+1, n_hid] -> [N, N, n_hid]
            mapped = torch.einsum('ijm,ml->ijl', corr, T_k)  # [N, N, n_hid]

            # Apply alpha: [N, N, n_hid] * [n_hid] -> [N, N]
            r_k = torch.einsum('ijl,l->ij', mapped, self.alpha)  # [N, N]

            # Apply LeakyReLU
            r_k = F.leaky_relu(r_k, negative_slope=0.01)

            # Store in r
            r[:, :, k] = r_k

        # Compute total directional influence strength S_{i->j}^t: [N, N]
        S = torch.sum(r, dim=2)  # [N, N]

        # Compute dynamic threshold
        mu_S = S.mean()  # Scalar: 1/(N^2) * sum(S_{i->j}^t)
        tau = self.beta * mu_S  # Scalar

        # Filter insignificant relationships
        mask = S > tau  # [N, N], boolean mask
        mask = mask.unsqueeze(-1)  # [N, N, 1], broadcastable to [N, N, (M+1)]

        # Apply softmax to the entire r tensor across the scale dimension (dim=2)
        R_softmax = F.softmax(r, dim=2)  # [N, N, (M+1)]

        # Initialize R and apply mask
        R = torch.zeros_like(r)  # [N, N, (M+1)]
        R[mask.expand_as(R)] = R_softmax[mask.expand_as(R_softmax)]
        # Where mask is False, R remains zero

        return R


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Memory(nn.Module):
    def __init__(self, num_memory, memory_dim, path_num):
        """
        Memory module for storing and retrieving risk propagation patterns.

        Args:
            num_memory (int): Number of memory units (Theta).
            memory_dim (int): Dimension of each memory unit (D, matches F).
            path_num (int): Number of propagation pathways (L).
        """
        super(Memory, self).__init__()
        self.num_memory = num_memory
        self.memory_dim = memory_dim
        self.path_num = path_num
        self.M = nn.Parameter(torch.randn(num_memory, memory_dim))
        self.keys = nn.Parameter(torch.randn(num_memory, memory_dim))
        self.values = nn.Parameter(torch.randn(num_memory, memory_dim))
        self.W_q = nn.Linear(2 * memory_dim, memory_dim, bias=False)
        self.W_p = nn.Linear(memory_dim, path_num, bias=True)

    def forward(self, h_i, e_j, neighbor_mask):
        """
        Retrieve risk pattern and compute path probabilities.

        Args:
            h_i (torch.Tensor): Node feature for stock i, shape [N, F].
            e_j (torch.Tensor): Reformed feature for stock j, shape [N, N, F].
            neighbor_mask (torch.Tensor): Neighbor mask, shape [N, N].

        Returns:
            torch.Tensor: Path probabilities [N, N, L].
        """
        N, _, Fea = e_j.shape
        q = self.W_q(torch.cat([h_i.unsqueeze(1).expand(-1, N, -1), e_j], dim=2))
        att_weight = F.softmax(torch.matmul(q, self.keys.t()) / math.sqrt(self.memory_dim), dim=-1)
        m_t = torch.matmul(att_weight, self.values)
        p = self.W_p(m_t)
        p = p.masked_fill(~neighbor_mask.unsqueeze(-1), float('-inf'))
        p = F.softmax(p, dim=2)
        return p


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Memory(nn.Module):
    def __init__(self, num_memory, memory_dim, path_num):
        super(Memory, self).__init__()
        self.num_memory = num_memory
        self.memory_dim = memory_dim
        self.path_num = path_num
        self.M = nn.Parameter(torch.randn(num_memory, memory_dim))
        self.keys = nn.Parameter(torch.randn(num_memory, memory_dim))
        self.values = nn.Parameter(torch.randn(num_memory, memory_dim))
        self.W_q = nn.Linear(2 * memory_dim, memory_dim, bias=False)
        self.W_p = nn.Linear(memory_dim, path_num, bias=True)

    def forward(self, h_i, e_j, neighbor_mask):
        """
        Retrieve risk pattern and compute path probabilities.

        Args:
            h_i (torch.Tensor): Node feature for stock i, shape [N, F].
            e_j (torch.Tensor): Reformed feature for stock j, shape [N, N, F].
            neighbor_mask (torch.Tensor): Neighbor mask, shape [N, N].

        Returns:
            torch.Tensor: Path probabilities [N, N, L].
        """
        N, _, Fea = e_j.shape
        q = self.W_q(torch.cat([h_i.unsqueeze(1).expand(-1, N, -1), e_j], dim=2))  # [N, N, memory_dim]
        att_weight = F.softmax(torch.matmul(q, self.keys.t()) / math.sqrt(self.memory_dim),
                               dim=-1)  # [N, N, num_memory]
        m_t = torch.matmul(att_weight, self.values)  # [N, N, memory_dim]
        p = self.W_p(m_t)  # [N, N, L]

        # 避免 nan：对无关系的股票对直接赋零概率
        p = p.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)  # 将无效邻居的路径概率置为 0
        p = F.softmax(p, dim=2)  # [N, N, L]
        p = p.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)  # 再次确保无效邻居概率为 0
        return p


class GATMechanism(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GATMechanism, self).__init__()
        self.W = nn.Linear(input_dim, output_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * output_dim))

    def forward(self, node_features, neighbor_mask):
        N = node_features.shape[0]
        h = self.W(node_features)  # [N, F']
        h_i = h.unsqueeze(1).expand(-1, N, -1)  # [N, N, F']
        h_j = h.unsqueeze(0).expand(N, -1, -1)  # [N, N, F']
        concat = torch.cat([h_i, h_j], dim=-1)  # [N, N, 2F']

        alpha = F.leaky_relu(torch.einsum('ijn,n->ij', concat, self.a), negative_slope=0.01)  # [N, N]
        alpha = alpha.masked_fill(~neighbor_mask, float('-inf'))
        valid_neighbors = neighbor_mask.any(dim=1, keepdim=True)  # [N, 1]
        alpha = alpha.masked_fill(~valid_neighbors, 0.0)  # 无邻居的节点置为 0
        alpha = F.softmax(alpha, dim=1)  # [N, N]
        alpha = alpha.masked_fill(~neighbor_mask, 0.0)  # 确保无效邻居权重为 0

        messages = alpha.unsqueeze(-1) * h_j  # [N, N, F']
        return messages


class SCGRN(nn.Module):
    def __init__(self, input_dim, scale_num, path_num, n_hid, num_memory=32):
        super(SCGRN, self).__init__()
        self.input_dim = input_dim
        self.scale_num = scale_num
        self.path_num = path_num
        self.n_hid = n_hid
        self.memory = Memory(num_memory, input_dim, path_num)
        self.gat_layers = nn.ModuleList([GATMechanism(input_dim, n_hid) for _ in range(path_num)])
        self.scale_encodings = self._get_scale_encodings()

    def _get_scale_encodings(self):
        encodings = []
        for m in range(self.scale_num):
            div_term = torch.exp(torch.arange(0, self.input_dim, 2).float() * (-math.log(10000.0) / self.input_dim))
            encoding = torch.zeros(self.input_dim)
            encoding[0::2] = torch.sin(torch.tensor(m, dtype=torch.float32) * div_term)[:self.input_dim // 2]
            encoding[1::2] = torch.cos(torch.tensor(m, dtype=torch.float32) * div_term)[:self.input_dim // 2]
            encodings.append(encoding)
        return torch.stack(encodings)

    def _compute_neighbors(self, R):
        neighbor_mask = torch.any(R != 0, dim=2)
        return neighbor_mask

    def forward(self, node_features, R):
        N, M, Fea = node_features.shape
        assert Fea == self.input_dim, f"Expected input_dim {self.input_dim}, got {Fea}"
        assert M == self.scale_num, f"Expected scale_num {self.scale_num}, got {M}"
        assert R.shape == (N, N, self.scale_num), f"Expected R shape {(N, N, self.scale_num)}, got {R.shape}"

        h_i = node_features[:, 0, :]  # [N, F]
        neighbor_mask = self._compute_neighbors(R)  # [N, N]
        scale_enc = self.scale_encodings.to(node_features.device)  # [M+1, F]
        h_j_m = node_features.unsqueeze(0)  # [1, N, M, F]
        weighted_features = torch.einsum('ijm,mf->ijmf', R, scale_enc) + h_j_m  # [N, N, M, F]
        e_j = weighted_features.sum(dim=2)  # [N, N, F]
        p = self.memory(h_i, e_j, neighbor_mask)  # [N, N, L]

        # GAT 消息传递
        messages = torch.zeros(N, N, self.n_hid, device=node_features.device)  # [N, N, F']
        for l in range(self.path_num):
            if p[..., l].sum() > 0:  # 仅对有效路径计算
                H_l = self.gat_layers[l](h_i, neighbor_mask)  # [N, N, F']
                messages += p[..., l].unsqueeze(-1) * H_l  # [N, N, F']

        v_tilde = F.leaky_relu(messages.sum(dim=1))  # [N, F']
        return v_tilde, p

