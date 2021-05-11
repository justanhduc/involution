//
// Created by justanhduc on 21. 5. 10..
//
#include <torch/torch.h>
#include <vector>

torch::Tensor involution_forward(const torch::Tensor input, const torch::Tensor weight, std::vector<int64_t> stride,
                                 std::vector<int64_t> padding, std::vector<int64_t> dilation);

torch::Tensor involution_backward_grad_input(torch::Tensor grads, torch::Tensor weight,
                                             std::vector<int64_t> input_shape, std::vector<int64_t> stride,
                                             std::vector<int64_t> padding, std::vector<int64_t> dilation);

torch::Tensor involution_backward_grad_weight(const torch::Tensor grads, const torch::Tensor input,
                                              std::vector<int64_t> weight_shape, std::vector<int64_t> stride,
                                              std::vector<int64_t> padding, std::vector<int64_t> dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("involution_forward", &involution_forward);
    m.def("involution_backward_grad_input", &involution_backward_grad_input);
    m.def("involution_backward_grad_weight", &involution_backward_grad_weight);
}
