#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F

from networks.b2f_modules.b2f_linear import B2FLinear
from networks.b2f_modules.b2f_relu import B2FReLU
from networks.b2f_modules.b2f_tanh import B2FTanh

# !!! For easy debugging, certain options are deleted when they are not used in the default DeepSDF specs

class B2FDecoder(nn.Module):
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
        super(B2FDecoder, self).__init__()

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
                
            # canceled weight norm
            setattr(self, "lin" + str(layer), B2FLinear(dims[layer], out_dim))

            # canceled batch norm

        self.relu = B2FReLU()
        self.tanh = B2FTanh()

    # input: N x (L+3)
    def forward(self, input):
        # input: [65536, 259].
        # 65536 = 4 * 16384 = ScenesPerBatch * SamplePerScene
        # 259 = 256 + 3 = latent + xyz

        xyz = input[:, -3:]
        
        print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] Entering decoder...')
        print(f'[HERE: In networks.deef_sdf_decoder.Decoder.forward] | input shape = {input.shape}...')

        # 9 layers in total: 0~8
        # - latent_in is for layer 4
        # - relu is for layer 0~7

        # forward propagation
        # layer 0
        lin0_out = self.lin0(input)
        relu0_out = self.relu(lin0_out)

        # layer 1
        lin1_out = self.lin1(relu0_out)
        relu1_out = self.relu(lin1_out)

        # layer 2
        lin2_out = self.lin2(relu1_out)
        relu2_out = self.relu(lin2_out)

        # layer 3
        lin3_out = self.lin3(relu2_out)
        relu3_out = self.relu(lin3_out)

        # layer 4
        # latent_in here
        cat4_out = torch.cat([relu3_out, input], 1) # [253 + 259]
        lin4_out = self.lin4(cat4_out)
        relu4_out = self.relu(lin4_out)

        # layer 5
        lin5_out = self.lin5(relu4_out)
        relu5_out = self.relu(lin5_out)

        # layer 6
        lin6_out = self.lin6(relu5_out)
        relu6_out = self.relu(lin6_out)

        # layer 7
        lin7_out = self.lin7(relu6_out)
        relu7_out = self.relu(lin7_out)

        # layer 8
        # not relu here
        lin8_out = self.lin8(relu7_out)
        
        # final tanh
        tanh_out = self.tanh(lin8_out) # [N_pts, 1]


        # b2f propagation
        # Note that tanh_out must be 1 dimensional except for the batching dimensions
        # Otherwise you need to create a unit matrix for the vectors, for grad_saved_output, the extra dimension has to go to the batching dimensions.

        # Note that, only for 1-dim outputs, final grad can be set to 'ones'.

        # b2f tanh
        grad_lin8_out = self.tanh.b2f_forward(saved_output=tanh_out, grad_saved_output=1.)

        # b2f layer 8
        grad_relu7_out = self.lin8.b2f_forward(grad_saved_output=grad_lin8_out)
        
        # b2f layer 7
        grad_lin7_out = self.relu.b2f_forward(saved_input=lin7_out, grad_saved_output=grad_relu7_out)
        grad_relu6_out = self.lin7.b2f_forward(grad_saved_output=grad_lin7_out)

        # b2f layer 6
        grad_lin6_out = self.relu.b2f_forward(saved_input=lin6_out, grad_saved_output=grad_relu6_out)
        grad_relu5_out = self.lin6.b2f_forward(grad_saved_output=grad_lin6_out)

        # b2f layer 5
        grad_lin5_out = self.relu.b2f_forward(saved_input=lin5_out, grad_saved_output=grad_relu5_out)
        grad_relu4_out = self.lin5.b2f_forward(grad_saved_output=grad_lin5_out)

        # b2f layer 4
        grad_lin4_out = self.relu.b2f_forward(saved_input=lin4_out, grad_saved_output=grad_relu4_out)
        grad_cat4_out = self.lin4.b2f_forward(grad_saved_output=grad_lin4_out)
        grad_relu3_out = grad_cat4_out[:, :253]
        grad_input_at_inserted = grad_cat4_out[:, 253:]

        # b2f layer 3
        grad_lin3_out = self.relu.b2f_forward(saved_input=lin3_out, grad_saved_output=grad_relu3_out)
        grad_relu2_out = self.lin3.b2f_forward(grad_saved_output=grad_lin3_out)

        # b2f layer 2
        grad_lin2_out = self.relu.b2f_forward(saved_input=lin2_out, grad_saved_output=grad_relu2_out)
        grad_relu1_out = self.lin2.b2f_forward(grad_saved_output=grad_lin2_out)

        # b2f layer 1
        grad_lin1_out = self.relu.b2f_forward(saved_input=lin1_out, grad_saved_output=grad_relu1_out)
        grad_relu0_out = self.lin1.b2f_forward(grad_saved_output=grad_lin1_out)

        # b2f layer 0
        grad_lin0_out = self.relu.b2f_forward(saved_input=lin0_out, grad_saved_output=grad_relu0_out)
        grad_input_at_start = self.lin0.b2f_forward(grad_saved_output=grad_lin0_out)

        # combining grad_input at start and at inserted
        grad_input = grad_input_at_start + grad_input_at_inserted # [N_pts, 256 + 3]


        return tanh_out, grad_input
