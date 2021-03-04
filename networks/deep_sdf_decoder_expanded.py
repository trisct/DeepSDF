#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


# !!! For easy debugging, certain options are deleted when they are not used in the default DeepSDF specs

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []
        
        print(f'[HERE: In networks.deef_sdf_decoder.Decoder] Initializing decoder...')

        dims = [latent_size + 3] + dims + [1] # [259, 512, ..., 512, 1], 10 in total
        self.num_layers = len(dims)

        print(f'[HERE: In networks.deef_sdf_decoder.Decoder] | dims = {dims}')
        print(f'[HERE: In networks.deef_sdf_decoder.Decoder] | num_layers = {self.num_layers}')
        # but in all below, layers go from 0 to num_layers-1 
        
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                
            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        # input: [65536, 259].
        # 65536 = 4 * 16384 = ScenesPerBatch * SamplePerScene
        # 259 = 256 + 3 = latent + xyz

        xyz = input[:, -3:]
        x = input
        
        print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] Entering decoder...')
        print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] | input shape = {input.shape}...')

        # 9 layers in total: 0~8
        # - latent_in is for layer 4
        # - relu is for layer 0~7

        # layer 0
        x = self.lin0(x)
        x = self.relu(x)

        # layer 1
        x = self.lin1(x)
        x = self.relu(x)

        # layer 2
        x = self.lin2(x)
        x = self.relu(x)

        # layer 3
        x = self.lin3(x)
        x = self.relu(x)

        # layer 4
        # latent_in here
        x = torch.cat([x, input], 1)
        x = self.lin4(x)
        x = self.relu(x)

        # layer 5
        x = self.lin5(x)
        x = self.relu(x)

        # layer 6
        x = self.lin6(x)
        x = self.relu(x)

        # layer 7
        x = self.lin7(x)
        x = self.relu(x)

        # layer 8
        # not relu here
        x = self.lin8(x)
        
        # final tanh
        x = self.th(x)

        return x
