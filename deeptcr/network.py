import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from deeptcr.modules import *
from deeptcr.network import RCCAModule


class CDREmbNet(nn.Module):
    def __init__(self, emb_size, vocab_size=33):
        super(CDREmbNet, self).__init__()
        self.emb_size = emb_size
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.a1_emb = nn.Embedding(vocab_size, emb_size)
        self.a2_emb = nn.Embedding(vocab_size, emb_size)
        self.a3_emb = nn.Embedding(vocab_size, emb_size)
        self.b1_emb = nn.Embedding(vocab_size, emb_size)
        self.b2_emb = nn.Embedding(vocab_size, emb_size)
        self.b3_emb = nn.Embedding(vocab_size, emb_size)
        self.reset_parameters()
        
    
    def forward(self, batch):        
        a1 = self.a1_emb(batch['A1'])
        a2 = self.a2_emb(batch['A2'])
        a3 = self.a3_emb(batch['A3'])
        b1 = self.b1_emb(batch['B1'])
        b2 = self.b2_emb(batch['B2'])
        b3 = self.b3_emb(batch['B3'])
        peptide = self.peptide_emb(batch['peptide'])
        emb_dict = {
            'A1': a1, 'A2': a2, 'A3': a3,
            'B1': b1, 'B2': b2, 'B3': b3,
            'peptide': peptide
        }
        return emb_dict

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -1, 1)
        nn.init.uniform_(self.a1_emb.weight, -1, 1)
        nn.init.uniform_(self.a2_emb.weight, -1, 1)
        nn.init.uniform_(self.a3_emb.weight, -1, 1)
        nn.init.uniform_(self.b1_emb.weight, -1, 1)
        nn.init.uniform_(self.b2_emb.weight, -1, 1)
        nn.init.uniform_(self.b3_emb.weight, -1, 1)
   
        
class Predictor(nn.Module):
    def __init__(self, c_in=512, c_hidden=64):
        super(Predictor, self).__init__()   
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.layer_norm = nn.LayerNorm(self.c_in)
        self.linear_1 = nn.Linear(self.c_in, self.c_hidden)
        self.linear_2 = nn.Linear(self.c_hidden, c_in)
        self.linear_3 = nn.Linear(self.c_in, 1)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        out = torch.sigmoid(s.flatten())
        return out    
            
    
class TECCNet(nn.Module):
    """
    TECC
    """
    def __init__(self, *, emb_size=20, hidden_size=64, vocab_size=33, dropout=0.1, emb_net=True):
        super(TECCNet, self).__init__()
        self.emb_net = emb_net
        self.emb_layer = CDREmbNet(emb_size, vocab_size)
        
        self.ccnet_a1 = RCCAModule(2*emb_size, hidden_size, dropout)
        self.ccnet_a2 = RCCAModule(2*emb_size, hidden_size, dropout)
        self.ccnet_a3 = RCCAModule(2*emb_size, hidden_size, dropout)
        self.ccnet_b1 = RCCAModule(2*emb_size, hidden_size, dropout)
        self.ccnet_b2 = RCCAModule(2*emb_size, hidden_size, dropout)
        self.ccnet_b3 = RCCAModule(2*emb_size, hidden_size, dropout)
        self.dropout = nn.Dropout2d(dropout)
        self.bindpredictor = Predictor(c_in=6*hidden_size, c_hidden=int(hidden_size/2))
    
    def build_map(self, seq_a, seq_b):        
        B, N, D2 = seq_a.shape
        B, M, D1 = seq_b.shape
        seq_b = seq_b.unsqueeze(2).expand(B, M, N, D1).contiguous()
        seq_a = seq_a.unsqueeze(1).expand(B, M, N, D2).contiguous()
        intermap = torch.cat([seq_b, seq_a], dim=-1)
        intermap = intermap.permute(0, 3, 1, 2).contiguous()
        return intermap   

    def forward(self, batch):
        if self.emb_net:
            emb_dict = self.emb_layer(batch)
            peptide_emb, a1_emb, a2_emb, a3_emb, b1_emb, b2_emb, b3_emb = emb_dict['peptide'], emb_dict['A1'], emb_dict['A2'], emb_dict['A3'], emb_dict['B1'], emb_dict['B2'], emb_dict['B3']
        else:
            peptide_emb, a1_emb, a2_emb, a3_emb, b1_emb, b2_emb, b3_emb = [
                batch[f'{key}_blosum50'] for key in 
            ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']
        ]
        
        intermap_a1 = self.build_map(a1_emb, peptide_emb)
        intermap_a1 = self.ccnet_a1(intermap_a1)
        intermap_a1 = reduce(intermap_a1, 'b d n m -> b d', 'max')
        intermap_a2 = self.build_map(a2_emb, peptide_emb)
        intermap_a2 = self.ccnet_a2(intermap_a2)
        intermap_a2 = reduce(intermap_a2, 'b d n m -> b d', 'max')
        intermap_a3 = self.build_map(a3_emb, peptide_emb)
        intermap_a3 = self.ccnet_a3(intermap_a3)
        intermap_a3 = reduce(intermap_a3, 'b d n m -> b d', 'max')
        
        intermap_b1 = self.build_map(b1_emb, peptide_emb)
        intermap_b1 = self.ccnet_b1(intermap_b1)
        intermap_b1 = reduce(intermap_b1, 'b d n m -> b d', 'max')
        intermap_b2 = self.build_map(b2_emb, peptide_emb)
        intermap_b2 = self.ccnet_b2(intermap_b2)
        intermap_b2 = reduce(intermap_b2, 'b d n m -> b d', 'max')
        intermap_b3 = self.build_map(b3_emb, peptide_emb)
        intermap_b3 = self.ccnet_b3(intermap_b3)
        intermap_b3 = reduce(intermap_b3, 'b d n m -> b d', 'max')
                
        s = torch.cat([intermap_a1, intermap_a2, intermap_a3, intermap_b1, intermap_b2, intermap_b3], dim=-1)
        
        bind_pred = self.bindpredictor(s)
        out = {
            'bind_pred':bind_pred, 
        }
        return out
    

class CNN_CDR123_global_max(nn.Module):
    """
    NetTCR-2.2 pan-specific
    """
    def __init__(self, *, dropout, embed_dim=20):
        super(CNN_CDR123_global_max, self).__init__()
        self.conv_activation = F.relu
        self.dense_activation = torch.sigmoid

        self.embed_dim = embed_dim

        # Define kernel sizes
        kernel_sizes = [k for k in range(1, 10, 2)]

        # Convolutional layers for each input
        self.pep_convs = self.create_conv_layers(kernel_sizes)
        self.a1_convs = self.create_conv_layers(kernel_sizes)
        self.a2_convs = self.create_conv_layers(kernel_sizes)
        self.a3_convs = self.create_conv_layers(kernel_sizes)
        self.b1_convs = self.create_conv_layers(kernel_sizes)
        self.b2_convs = self.create_conv_layers(kernel_sizes)
        self.b3_convs = self.create_conv_layers(kernel_sizes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Dense layers
        self.dense1 = nn.Linear(560, 64)  # 7 inputs * 5 kernel sizes * 16 filters each
        self.initialize_linear(self.dense1)  # Initialize dense layer
        self.dense2 = nn.Linear(64, 1)
        self.initialize_linear(self.dense2)  # Initialize dense layer

    def create_conv_layers(self, kernel_sizes):
        conv_layers = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k - 1) // 2
            conv = nn.Conv1d(self.embed_dim, 16, kernel_size=k, padding=padding)
            self.initialize_conv(conv)  # Initialize with Keras-like initialization
            conv_layers.append(conv)
        return conv_layers

    def initialize_conv(self, conv):
        # Keras-like 'glorot_uniform' initialization for Conv1D layers
        nn.init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.fill_(0.0)  # Keras default is zeros

    def initialize_linear(self, linear):
        # Keras-like 'glorot_uniform' initialization for Dense layers
        nn.init.xavier_uniform_(linear.weight)
        linear.bias.data.fill_(0.0)  # Keras default is zeros

    def forward(self, batch):
        # Unpack batch data
        pep, a1, a2, a3, b1, b2, b3 = [
            batch[f'{key}_blosum50'].transpose(-1, -2) for key in 
            ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']
        ]
        
        # Apply convolutions and global max pooling to each input
        pep_features = self.apply_convs(pep, self.pep_convs)
        a1_features = self.apply_convs(a1, self.a1_convs)
        a2_features = self.apply_convs(a2, self.a2_convs)
        a3_features = self.apply_convs(a3, self.a3_convs)
        b1_features = self.apply_convs(b1, self.b1_convs)
        b2_features = self.apply_convs(b2, self.b2_convs)
        b3_features = self.apply_convs(b3, self.b3_convs)

        # Concatenate all features
        cat_features = torch.cat([
            pep_features, a1_features, a2_features, a3_features,
            b1_features, b2_features, b3_features
        ], 1)

        # Pass through dense layers with dropout
        x = self.dropout(cat_features)
        x = self.dense_activation(self.dense1(x))
        out = torch.sigmoid(self.dense2(x).flatten())  # Apply sigmoid for binary classification

        return {'bind_pred': out}

    def apply_convs(self, x, convs):
        return torch.cat([F.max_pool1d(self.conv_activation(conv(x)), x.size(2)).squeeze(2) for conv in convs], 1)


class TransformerPredictor_AB_cdr123_with_epi(nn.Module):
    """
    MixTCRpred
    """
    def __init__(self, *, embedding_dim, hidden_dim, 
                 num_heads, num_layers, dropout, 
                 padding_idx, padding, vocab_size=33, num_labels=1):
        super(TransformerPredictor_AB_cdr123_with_epi, self).__init__()
        self.vocab_size =  vocab_size #q_aa
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.max_len = np.sum(padding)
        self.padding_idx = padding_idx

        
        #position wise embedding
        self.embedding_tok = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.scale = torch.sqrt(torch.FloatTensor([self.embedding_dim]))
        self.embedding_pos_epi = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[0])
        self.embedding_pos_cdr3_TRA = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1])
        self.embedding_pos_cdr3_TRB = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[2])
        self.embedding_pos = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, np.sum(self.padding[0:3]))
        self.embedding_pos_cdr12= PositionWiseEmbedding(self.vocab_size, self.embedding_dim, 4*self.padding[-1])
        ########## TEST ############3
        self.embedding_pos_TRA = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1] + 2*self.padding[-1])
        self.embedding_pos_TRB = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[2] + 2*self.padding[-1])
        # Transformer - Encoder
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers,
                                              input_dim=embedding_dim,
                                              dim_feedforward=hidden_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        ### Output classifier
        self.out_dim = np.sum(self.padding)*embedding_dim
        self.output_net= nn.Sequential(
            nn.Linear(self.out_dim,  embedding_dim),
            nn.LayerNorm(embedding_dim),
            # nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, num_labels)
            )
    def forward(self, batch, mask=True):
        if mask:
            mask_epi = (batch['peptide'] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRA = (batch['A3'] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRB = (batch['B3'] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr1_TRA = (batch['A1'] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr2_TRA = (batch['A2'] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr1_TRB = (batch['B1'] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr2_TRB = (batch['B2'] != self.padding_idx).unsqueeze(1).unsqueeze(2)
        else:
            mask_epi = None
            mask_cdr3_TRA = None
            mask_cdr3_TRB = None
            mask_cdr1_TRA = None
            mask_cdr2_TRA = None
            mask_cdr1_TRB = None
            mask_cdr2_TRB = None
        ####################################################################################################
        ### TRA
        ####################################################################################################
        ### cdr12_TRA
        x_cdr1_TRA = batch['A1']
        x_cdr2_TRA = batch['A2']
        x_cdr3_TRA = batch['A3']
        x_TRA = torch.concat([x_cdr1_TRA, x_cdr2_TRA, x_cdr3_TRA], dim = 1)
        mask_TRA = torch.concat([mask_cdr1_TRA, mask_cdr2_TRA, mask_cdr3_TRA], dim = 3)
        x_emb_TRA = self.embedding_tok(x_TRA)
        x_pos_TRA = self.embedding_pos_TRA(x_TRA)
        scale_TRA = self.scale.to(x_emb_TRA.device)
        x_TRA = (x_emb_TRA * scale_TRA) + x_pos_TRA
        x_TRA_out = self.transformer_encoder(x_TRA, mask=mask_TRA)
        ####################################################################################################
        ### TRB
        ####################################################################################################
        ### cdr12_TRB
        x_cdr1_TRB = batch['B1']
        x_cdr2_TRB = batch['B2']
        x_cdr3_TRB = batch['B3']
        x_TRB = torch.concat([x_cdr1_TRB, x_cdr2_TRB, x_cdr3_TRB], dim = 1)
        mask_TRB = torch.concat([mask_cdr1_TRB, mask_cdr2_TRB, mask_cdr3_TRB], dim = 3)
        x_emb_TRB = self.embedding_tok(x_TRB)
        x_pos_TRB = self.embedding_pos_TRB(x_TRB)
        scale_TRB = self.scale.to(x_emb_TRB.device)
        x_TRB = (x_emb_TRB * scale_TRB) + x_pos_TRB
        x_TRB_out = self.transformer_encoder(x_TRB, mask=mask_TRB)
        ####################################################################################################
        #EPITOPE
        ####################################################################################################
        x_epi = batch['peptide']
        x_emb_epi = self.embedding_tok(x_epi)
        #add positional embedding
        x_pos_epi = self.embedding_pos_epi(x_epi)
        scale_epi = self.scale.to(x_emb_epi.device)
        x_epi = (x_emb_epi * scale_epi) + x_pos_epi
        x_epi_out = self.transformer_encoder(x_epi, mask=mask_epi)
        ##x_epi = x_epi.flatten(start_dim = 1)
        #### FIX THE EPITOPE
        #x_epi_out = torch.zeros((x_epi.shape[0], x_epi.shape[1], x_TRA_out.shape[-1]), device = x_TRA.device)
        #concat
        x = torch.concat([x_epi_out, x_TRA_out, x_TRB_out], dim = 1)
        #add one last transformer encoder
        #x = self.transformer_encoder(x)#, mask=False)
        #flatten and classify
        x = x.flatten(start_dim = 1)
        x = self.output_net(x)
        return {'bind_pred': torch.sigmoid(x).flatten()}


