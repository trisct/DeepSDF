import torch
import torch.nn as nn
import torch.nn.functional as F

class B2FReLU(nn.Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Supports back propagation as b2f_forward.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(B2FReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=self.inplace)

    def b2f_forward(self, saved_input, grad_saved_output):
        """
        saved_input: the original input at the time of calling forward
        grad_saved_output: the gradient w.r.t. the original output as the time of calling forward

        The computes the grad w.r.t. the saved_input, according to the following rules:
        
            grad_saved_input[i] = grad_saved_output[i] * 1_{saved_input[i] >= 0}
        """

        grad_saved_input = torch.where(saved_input >= 0., grad_saved_output, torch.zeros_like(grad_saved_output, device=grad_saved_output.device))
        return grad_saved_input

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str + '. B2F version'
