#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


def pos_encoding(xyz):
    """
    xyz: [N, 3]
    ------
    output: [N, 3 * 20]
    """

    xyz = xyz.unsqueeze(dim=2).expand(-1, 3, 20)
    coef = torch.linspace(1., 10., 10, device=xyz.device).reshape(1, 10).expand(2, 10).reshape(1, 1, 20)
    coef_xyz = coef * xyz # [N, 3, 20]
    pos_enc_xyz = torch.cat((torch.sin(coef_xyz[:,:,0:10]), torch.cos(coef_xyz[:,:,10:20])), dim=2) # [N, 3, 20]
    pos_enc_xyz = pos_enc_xyz.reshape(-1, 60)
    return pos_enc_xyz


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
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3 + 60] + dims + [1] # 60 = 20 * 3 for positional encoding: x -> sin(x), cos(x), ..., sin(10x), cos(10x)

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in # what does this mean?
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.__init__] Displaying decoder attributes...')
        #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.__init__] | num_layers = %d' % self.num_layers)
        #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.__init__] | norm_layers =', self.norm_layers)
        #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.__init__] | latent_in =', self.latent_in)
        #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.__init__] | latent_dropout =', self.latent_dropout)

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in: # means that the latent code will be inserted here!
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

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]
        pos_enc_xyz = pos_encoding(xyz)

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input
        x = torch.cat([x, pos_enc_xyz], dim=1)

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.forward] layer = %d' % layer)
            #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.forward] lin =', lin)
            if layer in self.latent_in:
                #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.forward] layer is in latent_in, x cat with input. This means the latent code will be inserted here!')
                x = torch.cat([x, input, pos_enc_xyz], 1)
            elif layer != 0 and self.xyz_in_all:
                #print('[HERE: In networks.deep_sdf_decoder_pos_encode.Decoder.forward] layer != 0 and xyz_in_all x cat with xyz.')
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
