import torch
import torch.nn as nn
import torch.nn.functional as F

class B2FTanh(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Supports back propagation as b2f_forward.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input)

    def b2f_forward(self, saved_output, grad_saved_output):
        """
        saved_input: the original input at the time of calling forward
        grad_saved_output: the gradient w.r.t. the original output as the time of calling forward

        The computes the grad w.r.t. the saved_input, according to the following rules:
        
            First note that tanh^2(x) = 1 - tanh'(x). Hence, tanh'(x) = 1 - tanh^2(x). This results in

            grad_saved_input = grad_saved_output * (1 - saved_output ** 2)
        """

        grad_saved_input = grad_saved_output * (1 - saved_output ** 2)

        return grad_saved_input