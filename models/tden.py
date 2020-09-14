import os
import math
import torch
from torch import nn
import torch.nn.functional as F

import bert.modeling_bert as bert
import models.basic_layer as basic_layer

from bert.activations import gelu, gelu_new, swish, mish
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import random
import numpy as np
from lib.config import cfg

class TDEN(nn.Module):
    def __init__(self, config):
        super(TDEN, self).__init__()
        self.visn_fc = basic_layer.BertVEmbeddings(config)

        self.layer = nn.ModuleList(
            [bert.BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.v_layer = nn.ModuleList(
            [basic_layer.BertImageLayer(config) for _ in range(config.v_num_hidden_layers)]
        )

        if cfg.MODEL.USE_DECODER == True:
            self.d_layer = nn.ModuleList(
                [basic_layer.DecoderLayer(config) for _ in range(config.dec_num_hidden_layers)]
            )

        if cfg.MODEL.USE_CROSSER == True:
            self.x_layer = nn.ModuleList(
                [basic_layer.BertConnectionLayer(config) for _ in range(config.enc_num_hidden_layers) ]
            )


    def forward(self, lang_feats, v_feats, lang_attention_mask=None, v_attention_mask=None):
        v_feats = self.visn_fc(v_feats)

        # Run language layers
        lang_feats_arr = []
        for layer_module in self.layer:
            lang_feats, _ = layer_module(lang_feats, lang_attention_mask)
            lang_feats_arr.append(lang_feats)

        # Run vision layers
        v_feats_arr = []
        for layer_module in self.v_layer:
            v_feats, _ = layer_module(v_feats, v_attention_mask)
            v_feats_arr.append(v_feats)

        return lang_feats_arr, v_feats_arr

    def forward_enc_dec(self, lang_feats_u, lang_feats_g, v_feats, lang_attention_mask_u=None, lang_attention_mask_g=None, v_attention_mask=None):
        v_feats = self.visn_fc(v_feats)

        # Run language layers
        lang_feats_arr_u = []
        for layer_module in self.layer:
            lang_feats_u, _ = layer_module(lang_feats_u, lang_attention_mask_u)
            lang_feats_arr_u.append(lang_feats_u)

        lang_feats_arr_g = []
        for layer_module in self.layer:
            lang_feats_g, _ = layer_module(lang_feats_g, lang_attention_mask_g)
            lang_feats_arr_g.append(lang_feats_g)

        # Run vision layers
        v_feats_arr = []
        for layer_module in self.v_layer:
            v_feats, _ = layer_module(v_feats, v_attention_mask)
            v_feats_arr.append(v_feats)

        return lang_feats_arr_u, lang_feats_arr_g, v_feats_arr

    def forward_v(self, v_feats, v_attention_mask=None):
        v_feats = self.visn_fc(v_feats)

        # Run vision layers
        v_feats_arr = []
        for layer_module in self.v_layer:
            v_feats, _ = layer_module(v_feats, v_attention_mask)
            v_feats_arr.append(v_feats)

        return v_feats_arr

    def forward_t(self, lang_feats, lang_attention_mask=None):
        # Run language layers
        lang_feats_arr = []
        for layer_module in self.layer:
            lang_feats, _ = layer_module(lang_feats, lang_attention_mask)
            lang_feats_arr.append(lang_feats)

        return lang_feats_arr


    # understanding
    def uforward(self, lang_feats_arr, v_feats_arr, lang_attention_mask=None, v_attention_mask=None):
        lang_feats = lang_feats_arr[-1]
        v_feats = v_feats_arr[-1]

        nv_feats_arr = []
        nlang_feats_arr = []
        for i, layer_module in enumerate(self.x_layer):
            v_feats, lang_feats = layer_module(v_feats, v_attention_mask, lang_feats, lang_attention_mask)
            nv_feats_arr.append(v_feats)
            nlang_feats_arr.append(lang_feats)

        return nlang_feats_arr, nv_feats_arr

    # generation
    def gforward(self, lang_feats_arr, v_feats_arr, lang_attention_mask=None, v_attention_mask=None):
        lang_feats = lang_feats_arr[-1]
        for i, layer_module in enumerate(self.d_layer):
            lang_feats = layer_module(lang_feats, v_feats_arr, lang_attention_mask, v_attention_mask)

        return lang_feats

    def decode(self, lang_feats, v_feats, lang_attention_mask=None, v_attention_mask=None, t_history_states=None, v_history_states=None):
        if v_history_states is None:
            v_history_states = [None] * (len(self.v_layer) + 1)
        if t_history_states is None:
            t_history_states = [None] * (len(self.layer) + len(self.d_layer))
        
        if v_history_states[0] is not None:
            v_feats = v_history_states[0]
        else:
            v_feats = self.visn_fc(v_feats)
            v_history_states[0] = v_feats

        # Run vision layers
        v_feats_arr = []
        for i, layer_module in enumerate(self.v_layer):
            if v_history_states[i+1] is not None:
                v_feats = v_history_states[i+1]
            else:
                v_feats, _ = layer_module(v_feats, v_attention_mask)
                v_history_states[i+1] = v_feats
            v_feats_arr.append(v_feats)

        # Run language layers
        for i, layer_module in enumerate(self.layer):
            if t_history_states[i] is None:
                t_history_states[i] = lang_feats
            else:
                t_history_states[i] = torch.cat([t_history_states[i], lang_feats], dim=1)
            lang_feats, _ = layer_module(lang_feats, lang_attention_mask, t_history_states[i])

        for i, layer_module in enumerate(self.d_layer):
            if t_history_states[i + len(self.layer)] is None:
                t_history_states[i + len(self.layer)] = lang_feats
            else:
                t_history_states[i + len(self.layer)] = torch.cat([t_history_states[i + len(self.layer)], lang_feats], dim=1)

            lang_feats = layer_module(lang_feats, v_feats_arr, lang_attention_mask, v_attention_mask, t_history_states[i + len(self.layer)])

        return lang_feats, t_history_states, v_history_states

