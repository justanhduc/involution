//
// Created by justanhduc on 21. 5. 10..
//

#ifndef INVOLUTION_UTILS_H
#define INVOLUTION_UTILS_H

#include <torch/extension.h>
#include <torch/torch.h>

#define CHECK_CUDA(x) \
TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_FLOAT(x) \
TORCH_CHECK((x.scalar_type() == torch::kFloat32) || \
(x.scalar_type() == torch::kFloat16) ||             \
(x.scalar_type() == torch::kFloat64), #x " must be a half, float or double tensor")

#define CHECK_INPUT(x) \
CHECK_CUDA(x);         \
CHECK_CONTIGUOUS(x);   \
CHECK_FLOAT(x)

#endif //INVOLUTION_UTILS_H
