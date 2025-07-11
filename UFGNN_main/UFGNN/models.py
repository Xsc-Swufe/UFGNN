''' Define the HGTAN model '''
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
import torch.nn as nn
from training.tools import *
import torch.nn.functional as F
from UFGNN.layers import *
import torch.fft


class UFGNN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            num_stock, rnn_unit, n_hid, n_class,
            feature,
            d_word_vec, d_model, dropout,
            tgt_emb_prj_weight_sharing,scale_num,path_num,window_size):

        super().__init__()
        self.dropout = dropout
        self.linear = nn.Linear(feature, d_word_vec)
        self.scale_num = scale_num
        self.path_num = path_num
        self.n_hid=n_hid
        # torch.nn.init.xavier_uniform_(self.linear.weight)
        # if self.linear.bias is not None:
        #     torch.nn.init.zeros_(self.linear.bias)
        self.mstdm = MSTDM(
            input_dim=feature,
            scale_num=scale_num,
            dropout=dropout,
            rnn_unit=rnn_unit
        )

        self.tmirim = TMIRIM(
            input_dim=rnn_unit,
            scale_num=scale_num,
            n_hid=n_hid,
            beta=1
        )

        # self.scgrn = SCGRN(
        #     input_dim=rnn_unit,
        #     scale_num=scale_num,
        #     path_num=path_num,
        #     n_hid=n_hid
        # )

        self.scgrn = SCGRN(input_dim=rnn_unit, scale_num=scale_num, path_num=path_num, n_hid=n_hid, num_memory=10)


        self.tgt_word_prj = nn.Linear(2*n_hid, n_class, bias=False)

        self.linear = nn.Linear(scale_num*rnn_unit, n_hid, bias=False)
        #self.tgt_word_prj = nn.Linear(2*rnn_unit, n_class, bias=False)
        #self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        self.rnn2 = nn.GRU(feature,
                           rnn_unit,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
        #self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        #nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1



        self.ln = nn.LayerNorm(rnn_unit)
        self.ln2 = nn.LayerNorm(rnn_unit * scale_num)
        self.ln_in = nn.LayerNorm(feature)
        self.ln3 = nn.LayerNorm(2*n_hid)
        self.ln5 = nn.LayerNorm(n_hid)
        #self.ln2 = nn.LayerNorm(rnn_unit * scale_num)

        #self.mstdm = MultiScaleTemporalDecompositionModule(T=T, C=C, K=3, output_feature_dim=64)
    def forward(self,src_seq):
        stock_num = src_seq.size(1)
        seq_len = src_seq.size(0)
        dim = src_seq.size(2)

        if torch.isnan(src_seq).any():
            print("src_seq1 中存在 NaN 值！")




        src_seq=src_seq
        new_src_seq = self.mstdm(src_seq)   #input T N C  output N scale_num F
        new_src_seq = F.dropout(new_src_seq, self.dropout, training=self.training)

        new_src_seq_reshaped = new_src_seq.reshape(stock_num, self.scale_num * new_src_seq.shape[-1])  #output N scale_num*F

        new_src_seq_reshaped = self.linear(new_src_seq_reshaped)
        #new_src_seq_reshaped = F.dropout(new_src_seq_reshaped, self.dropout, training=self.training)
        #new_src_seq_reshaped = self.ln5(new_src_seq_reshaped)
        rel_R = self.tmirim(new_src_seq)  #171 171 3

        #output = self.scgrn(new_src_seq[:,0,:], rel_R)
        output, p = self.scgrn(new_src_seq, rel_R)  #   N    F'

        output = F.dropout(output, self.dropout, training=self.training)


        combined_output = torch.cat((new_src_seq_reshaped, output), dim=1)
        #combined_output = self.ln3(combined_output)

        #seq_logit = F.elu(self.tgt_word_prj(output))
        seq_logit = F.elu(self.tgt_word_prj(combined_output))

        output = F.log_softmax(seq_logit, dim=1)


        return output, p


