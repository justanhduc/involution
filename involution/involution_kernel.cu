//
// Created by justanhduc on 21. 5. 10..
//
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "utils.h"

#define NUM_THREADS 1024


template <typename scalar_t>
__global__ void involution_forward_kernel(const scalar_t* bottom_data, const scalar_t* weight_data, scalar_t* top_data,
                                          int bs, int channels, int bottom_height, int bottom_width,
                                          int top_height, int top_width, int kernel_h, int kernel_w,
                                          int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
                                          int groups) {
    const int n = bs * channels * top_height * top_width;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const int n = index / channels / top_height / top_width;
        const int c = (index / top_height / top_width) % channels;
        const int h = (index / top_width) % top_height;
        const int w = index % top_width;
        const int g = c / (channels / groups);
        scalar_t value = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_in = -pad_h + h * stride_h + kh * dilation_h;
                const int w_in = -pad_w + w * stride_w + kw * dilation_w;
                if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
                    const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
                    const int offset_weight = ((((n * groups + g) * kernel_h + kh) * kernel_w + kw) * top_height + h) * top_width + w;
                    value += weight_data[offset_weight] * bottom_data[offset];
                }
            }
        }
        top_data[index] = value;
    }
}


torch::Tensor involution_forward(const torch::Tensor input, const torch::Tensor weight, std::vector<int64_t> stride,
                                 std::vector<int64_t> padding, std::vector<int64_t> dilation) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    const auto bs = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    const auto output_h = (height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1;
    const auto output_w = (width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1;

    auto output = torch::empty({bs, channels, output_h, output_w}, input.options());

    const int NUM_BLOCKS = (bs * channels * output_h * output_w + NUM_THREADS - 1) / NUM_THREADS;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "involution_forward_kernel", [&] {
        involution_forward_kernel<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            bs, channels, height, width, output_h, output_w, kernel_h, kernel_w,
            padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1], weight.size(1)
        );
    });

    return output;
}


template <typename scalar_t>
__global__ void involution_backward_grad_input_kernel(const scalar_t* const top_diff, const scalar_t* const weight_data,
                                                      scalar_t* const bottom_diff, int bs, int channels,
                                                      int bottom_height, int bottom_width,
                                                      int top_height, int top_width, int kernel_h, int kernel_w,
                                                      int pad_h, int pad_w, int stride_h, int stride_w,
                                                      int dilation_h, int dilation_w, int groups) {
    const int n = bs * channels * bottom_height * bottom_width;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const int n = index / channels / bottom_height / bottom_width;
        const int c = (index / bottom_height / bottom_width) % channels;
        const int h = (index / bottom_width) % bottom_height;
        const int w = index % bottom_width;
        const int g = c / (channels / groups);
        scalar_t value = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_out_s = h + pad_h - kh * dilation_h;
                const int w_out_s = w + pad_w - kw * dilation_w;
                if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
                    const int h_out = h_out_s / stride_h;
                    const int w_out = w_out_s / stride_w;
                    if ((h_out >= 0) && (h_out < top_height)
                    && (w_out >= 0) && (w_out < top_width)) {
                        const int offset = ((n * channels + c) * top_height + h_out) * top_width + w_out;
                        const int offset_weight = ((((n * groups + g) * kernel_h + kh) * kernel_w + kw) *
                                top_height + h_out) * top_width + w_out;
                        value += weight_data[offset_weight] * top_diff[offset];
                    }
                }
            }
        }
        bottom_diff[index] = value;
    }
}


torch::Tensor involution_backward_grad_input(const torch::Tensor grads, const torch::Tensor weight,
                                             std::vector<int64_t> input_shape, std::vector<int64_t> stride,
                                             std::vector<int64_t> padding, std::vector<int64_t> dilation) {
    CHECK_INPUT(grads);
    CHECK_INPUT(weight);

    const auto bs = input_shape[0];
    const auto channels = input_shape[1];
    const auto height = input_shape[2];
    const auto width = input_shape[3];
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    const auto output_h = grads.size(2);
    const auto output_w = grads.size(3);

    auto grad_input = torch::empty(input_shape, grads.options());

    const int NUM_BLOCKS = (bs * channels * height * width + NUM_THREADS - 1) / NUM_THREADS;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grads.scalar_type(), "involution_backward_grad_input_kernel", [&] {
        involution_backward_grad_input_kernel<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            grads.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            bs, channels, height, width, output_h, output_w, kernel_h, kernel_w,
            padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1], weight.size(1)
        );
    });

    return grad_input;
}


template <typename scalar_t>
__global__ void involution_backward_grad_weight_kernel(const scalar_t* const top_diff, const scalar_t* const bottom_data,
                                                       scalar_t* const buffer_data, int bs, int channels,
                                                       int bottom_height, int bottom_width,
                                                       int top_height, int top_width, int kernel_h, int kernel_w,
                                                       int pad_h, int pad_w, int stride_h, int stride_w,
                                                       int dilation_h, int dilation_w, int groups) {
    const int n = bs * groups * kernel_h * kernel_w * top_height * top_width;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        const int h = (index / top_width) % top_height;
        const int w = index % top_width;
        const int kh = (index / kernel_w / top_height / top_width) % kernel_h;
        const int kw = (index / top_height / top_width) % kernel_w;
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height)
        && (w_in >= 0) && (w_in < bottom_width)) {
            const int g = (index / kernel_h / kernel_w / top_height / top_width) % groups;
            const int n = (index / groups / kernel_h / kernel_w / top_height / top_width) % bs;
            scalar_t value = 0;
            for (int c = g * (channels / groups); c < (g + 1) * (channels / groups); ++c) {
                const int top_offset = ((n * channels + c) * top_height + h) * top_width + w;
                const int bottom_offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
                value += top_diff[top_offset] * bottom_data[bottom_offset];
            }
            buffer_data[index] = value;
        } else {
            buffer_data[index] = 0;
        }
    }
}


torch::Tensor involution_backward_grad_weight(const torch::Tensor grads, const torch::Tensor input,
                                              std::vector<int64_t> weight_shape, std::vector<int64_t> stride,
                                              std::vector<int64_t> padding, std::vector<int64_t> dilation) {
    CHECK_INPUT(input);
    CHECK_INPUT(grads);

    const auto bs = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto kernel_h = weight_shape[2];
    const auto kernel_w = weight_shape[3];

    const auto output_h = grads.size(2);
    const auto output_w = grads.size(3);

    auto grad_weight = torch::empty(weight_shape, grads.options());

    const int NUM_BLOCKS = (bs * weight_shape[1] * kernel_h * kernel_w * output_h * output_w + NUM_THREADS - 1) / NUM_THREADS;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grads.scalar_type(), "involution_backward_grad_weight_kernel", [&] {
        involution_backward_grad_weight_kernel<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            grads.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), grad_weight.data_ptr<scalar_t>(),
            bs, channels, height, width, output_h, output_w, kernel_h, kernel_w,
            padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1], weight_shape[1]
        );
    });

    return grad_weight;
}
