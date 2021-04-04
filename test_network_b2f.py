import torch
import torch.nn

from networks.deep_sdf_decoder_expanded import B2FDecoder

b2f_decoder = B2FDecoder(
                    latent_size=256,
                    dims=[512, 512, 512, 512, 512, 512, 512, 512],
                    latent_in=[4])

input = torch.randn(2000, 259, requires_grad=True)

out, grad_input = b2f_decoder(input)


i = 9
out[i].sum().backward()
print(f'In input.grad[{i}] is equal to grad_input[{i}] at {(input.grad[i] == grad_input[i]).sum()}/{grad_input[i].shape[0]} places')
