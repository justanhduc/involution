from .involution import Involution
from .involution_naive import Involution as InvolutionNaive

import torch
import pytest


@pytest.mark.parametrize('dtype', (torch.float16, torch.float32, torch.float64))
@pytest.mark.parametrize('k_size', (3, 5))
@pytest.mark.parametrize('stride', (1, 2))
@pytest.mark.parametrize('dilation', (1, 2))
def test(dtype, k_size, stride, dilation):
    input = torch.randn(8, 16, 32, 32).cuda().to(dtype).requires_grad_(True)
    inv = Involution(input.shape[1], k_size, stride, dilation).to(dtype)
    inv_naive = InvolutionNaive(input.shape[1], k_size, stride).to(dtype)
    for p, pn in zip(inv.parameters(), inv_naive.parameters()):
        pn.data = p.data.clone()

    inv.cuda()
    inv_naive.cuda()
    out = inv(input)
    out_naive = inv_naive(input)
    torch.testing.assert_allclose(out, out_naive)

    loss = torch.mean(out ** 2)
    loss.backward(retain_graph=True)
    grad = input.grad.clone()

    input.grad *= 0.
    loss_naive = torch.mean(out_naive ** 2)
    loss_naive.backward()
    torch.testing.assert_allclose(grad, input.grad)
