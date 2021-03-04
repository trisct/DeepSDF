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
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

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
        xyz = input[:, -3:]

        x = input

        print('[HERE: In networks.deef_sdf_decoder.Decoder.forward] Passing through decoder...')
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] | At layer {layer}: Got lin_layer: {lin}')
            
            if layer in self.latent_in:
                print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] | | latent_in specified here. Feature shape is (before concat) = {x.shape}')
                x = torch.cat([x, input], 1)
                print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] | | latent_in specified here. Feature shape is (after concat) = {x.shape}')
            
            x = lin(x)

            if layer < self.num_layers - 2:
                print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] | | Applying relu here. Feature shape is = {x.shape}')
                x = self.relu(x)

        print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] All layers passed.')
        if hasattr(self, "th"):
            print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] decoder has "th" attribute. Applying tanh')
            x = self.th(x)

        return x
