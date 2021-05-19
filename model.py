import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from torchcrf import CRF
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x = x + self.scale * self.pe[:x.size(0), :]
        x += self.pe[:x.size(0), :]

        return self.dropout(x)



class Encoder(nn.Module):
    def __init__(self, input_size, n_head_encoder, d_model, n_layers_encoder, max_len):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.n_head_encoder = n_head_encoder
        self.d_model = d_model
        self.n_layers_encoder = n_layers_encoder
        self.max_len = max_len

        # define the layers
        self.lin_transform = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_len)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.n_head_encoder, self.d_model*4, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_layers_encoder)


    def forward(self, input_tensor, mask, src_key_padding_mask):
        input_tensor = self.lin_transform(input_tensor)
        input_tensor *= math.sqrt(self.d_model)
        input_encoded = self.pos_encoder(input_tensor)
        output = self.transformer_encoder(input_encoded, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output




class Transformer(nn.Module):
    def __init__(self, input_size, n_tokens, n_head_encoder, d_model, n_layers_encoder, max_len):
        super(Transformer, self).__init__()
         
        self.encoder = Encoder(input_size, n_head_encoder, d_model, n_layers_encoder, max_len)
        self.blstm = nn.LSTM(d_model,
                            d_model,
                            num_layers=2,
                            bidirectional=True
                            )

        #self.blstm = nn.LSTM(40,
        #                    d_model,
        #                    num_layers=4,
        #                    bidirectional=True
        #                    )
 
        self.out = nn.Linear(d_model, n_tokens)
   

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
    def generate_subsequent_mask(self, seq):
        len_s = seq.size(0)
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask


    def create_mask(self, src):
        src_seq_len = src.shape[0]

        src_mask = self.generate_square_subsequent_mask(src_seq_len).to(device=src.device)

        src_padding_mask = (src == 0).transpose(0, 1)
        return src_mask, src_padding_mask[:, :, 0]


    def forward(self, input_seq, input_seq_lengths, input_sent_seq, input_sent_lengths):
        src_mask, src_padding_mask = self.create_mask(input_seq)
        
        memory = self.encoder(input_seq, mask=src_mask, src_key_padding_mask=src_padding_mask)
        output, hidden = self.blstm(memory)
        hidden = torch.mean(hidden[0], dim=0)
        output = self.out(hidden)
        
        # process just the text
        #packed_seq = pack_padded_sequence(input_sent_seq, input_sent_lengths, enforce_sorted=False)
        #output, hidden = self.blstm(packed_seq)
        #output = pad_packed_sequence(output)[0]
        #
        #hidden = torch.mean(hidden[0], dim=0)
        #output = self.out(hidden)

        return output




