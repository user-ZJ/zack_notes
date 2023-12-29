libtorch使用
=====================

下载
------------
https://pytorch.org/get-started/locally/

需要安装对应版本的cuda

如果你已经安装过Pytorch，那么就不用额外安装Libtorch了，因为Pytorch自带了Libtorch的CMake config 文件，使用torch.utils.cmake_prefix_path语句就能打印出来

python -c 'import torch;print(torch.utils.cmake_prefix_path)'


示例
---------------------------------
.. code-block:: cpp

    #include <iostream>
    #include <torch/torch.h>

    int main() {
        std::cout << torch::cuda::is_available() << std::endl;
        std::cout << torch::cuda::cudnn_is_available() << std::endl;
        std::cout << torch::cuda::device_count() << std::endl;
        // 使用arange构造一个一维向量，再用reshape变换到5x5的矩阵
        torch::Tensor foo = torch::arange(25).reshape({5, 5}).to(torch::kCUDA);

        // 计算矩阵的迹
        torch::Tensor bar = torch::einsum("ii", foo);

        // 输出矩阵和对应的迹
        std::cout << "==> matrix is:\n " << foo << std::endl;
        std::cout << "==> trace of it is:\n " << bar << std::endl;
    }

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
    project(libtorchDemo)

    set(CMAKE_CXX_STANDARD 17)
    set(Torch_DIR /root/to/libtorch/share/cmake/Torch)

    find_package(CUDA REQUIRED)
    find_package(Torch REQUIRED)

    include_directories(
        /usr/local/cuda/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        libtorch/include
        libtorch/include/torch/csrc/api/include
    )

    cuda_add_executable(libtorch_test libtorch_test.cpp)
    target_link_libraries(libtorch_test "${TORCH_LIBRARIES}")


tensor操作
-------------------

.. code-block:: cpp

    #include <torch/torch.h>
    #include <iostream>
    // 使用固定维度，指定值初始化tensor
    torch::Tensor b = torch::zeros({3,4});
    b = torch::ones({3,4});
    b= torch::eye(4);  // 对角线全为1，其余全为0
    b = torch::full({3,4},10);
    b = torch::tensor({33,22,11});
    // 固定维度，随机值初始化
    torch::Tensor r = torch::rand({3,4});
    r = torch::randn({3, 4});
    r = torch::randint(0, 4,{3,3}); // 0-4范围的值初始化维度为3x3
    // C++中数据结构初始化tensor
    int aa[10] = {3,4,6};
    std::vector<float> aaaa = {3,4,6};
    auto aaaaa = torch::from_blob(aa,{3},torch::kFloat);
    auto aaa = torch::from_blob(aaaa.data(),{3},torch::kFloat);
    std::cout << aaa << std::endl;
    // 对已经创建的tensor进行初始化
    auto b = torch::zeros({3,4});
    auto d = torch::Tensor(b);
    d = torch::zeros_like(b);
    d = torch::ones_like(b);
    d = torch::rand_like(b,torch::kFloat);
    d = b.clone();
    // 存虚数的tensor
    torch::Tensor tt = torch::rand({2,3},torch::kComplexFloat);
    auto accessor = tt.accessor<c10::complex<float>, 2>();
    for (int i = 0; i < stft.size(0); i++) {
        for (int j = 0; j < stft.size(1); j++) {
            std::cout << accessor[i][j].real() << "+" << accessor[i][j].imag() << "i ";
        }
        std::cout << std::endl;
    }


    // 改变tensor的维度
    auto b = torch::full({10},3);
    b.view({1, 2,-1});
    std::cout<<b;
    b = b.view({1, 2,-1});
    std::cout<<b;
    auto c = b.transpose(0,1);
    std::cout<<c;
    auto d = b.reshape({1,1,-1});
    std::cout<<d;
    auto e = b.permute({1,0,2});
    std::cout<<e;

    // 切片
    auto b = torch::rand({10,3,28,28});//BxCxHxW
    std::cout<<b[0].sizes();//0th picture
    std::cout<<b[0][0].sizes();//0th picture, 0th channel
    std::cout<<b[0][0][0].sizes();//0th picture, 0th channel, 0th row pixels
    std::cout<<b[0][0][0][0].sizes();//0th picture, 0th channel, 0th row, 0th column pixels
    std::cout<<b.index_select(0,torch::tensor({0, 3, 3})).sizes();//choose 0th dimension at 0,3,3 to form a tensor of [3,3,28,28]
    std::cout<<b.index_select(1,torch::tensor({0,2})).sizes(); //choose 1th dimension at 0 and 2 to form a tensor of[10, 2, 28, 28]
    std::cout<<b.index_select(2,torch::arange(0,8)).sizes(); //choose all the pictures' first 8 rows [10, 3, 8, 28]
    std::cout<<b.narrow(1,0,2).sizes();//choose 1th dimension, from 0, cutting out a lenth of 2, [10, 2, 28, 28]
    std::cout<<b.select(3,2).sizes();//select the second tensor of the third dimension, that is, the tensor composed of the second row of all pictures [10, 3, 28]
    auto c = torch::randn({3,4});
    auto mask = torch::zeros({3,4});
    mask[0][0] = 1;
    std::cout<<c;
    std::cout<<c.index({mask.to(torch::kBool)});
    auto c = torch::randn({ 3,4 });
    auto mask = torch::zeros({ 3,4 });
    mask[0][0] = 1;
    mask[0][2] = 1;
    std::cout << c;
    std::cout << c.index({ mask.to(torch::kBool) });
    std::cout << c.index_put_({ mask.to(torch::kBool) }, c.index({ mask.to(torch::kBool) })+1.5);
    std::cout << c;

    // tensor操作
    auto b = torch::ones({3,4});
    auto c = torch::zeros({3,4});
    auto cat = torch::cat({b,c},1);//1 refers to 1th dim, output a tensor of shape [3,8]
    auto stack = torch::stack({b,c},1);//1refers to 1th dim, output a tensor of shape [3,2,4]
    std::cout<<b<<c<<cat<<stack;
    auto b = torch::rand({3,4});
    auto c = torch::rand({3,4});
    // mul div mm bmm
    std::cout<<b<<c<<b*c<<b/c<<b.mm(c.t());




自定义C++/cuda算子
-------------------------------------
https://pytorch.org/tutorials/advanced/cpp_extension.html

以LLTM为例，实现自定义的kernel

接口定义
`````````````````
.. code-block:: cpp

    // lltm_cuda.h
    #include <torch/torch.h>
    #include <vector>

    std::vector<torch::Tensor> lltm_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell);

    std::vector<torch::Tensor> lltm_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gate_weights,
        torch::Tensor weights);

.. code-block:: cpp

    // lltm_cuda.cpp
    #include <torch/torch.h>
    #include <vector>
    #include "lltm_cuda.h"

    std::vector<torch::Tensor> lltm_cuda_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell);

    std::vector<torch::Tensor> lltm_cuda_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gate_weights,
        torch::Tensor weights);

    // C++ interface

    #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
    #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
    #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

    std::vector<torch::Tensor> lltm_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(old_h);
    CHECK_INPUT(old_cell);

    return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
    }

    std::vector<torch::Tensor> lltm_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gate_weights,
        torch::Tensor weights) {
    CHECK_INPUT(grad_h);
    CHECK_INPUT(grad_cell);
    CHECK_INPUT(input_gate);
    CHECK_INPUT(output_gate);
    CHECK_INPUT(candidate_cell);
    CHECK_INPUT(X);
    CHECK_INPUT(gate_weights);
    CHECK_INPUT(weights);

    return lltm_cuda_backward(
        grad_h,
        grad_cell,
        new_cell,
        input_gate,
        output_gate,
        candidate_cell,
        X,
        gate_weights,
        weights);
    }

.. code-block:: cpp

    // lltm_cuda_kernel.cu
    #include <vector>
    #include <torch/torch.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
        return 1.0 / (1.0 + exp(-z));
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
    const auto s = sigmoid(z);
        return (1.0 - s) * s;
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
    const auto t = tanh(z);
        return 1 - (t * t);
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
        return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
        const auto e = exp(z);
        const auto d_relu = z < 0.0 ? 0.0 : 1.0;
        return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
    }

    template <typename scalar_t>
    __global__ void lltm_cuda_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
        //batch index
        const int n = blockIdx.y;
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < gates.size(2)){
            input_gate[n][c] = sigmoid(gates[n][0][c]);
            output_gate[n][c] = sigmoid(gates[n][1][c]);
            candidate_cell[n][c] = elu(gates[n][2][c]);
            new_cell[n][c] =
                old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
            new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
        }
    }

    std::vector<torch::Tensor> lltm_cuda_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell) {
        auto X = torch::cat({old_h, input}, /*dim=*/1);
        auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

        const auto batch_size = old_cell.size(0);
        const auto state_size = old_cell.size(1);

        auto gates = gate_weights.reshape({batch_size, 3, state_size});
        auto new_h = torch::zeros_like(old_cell);
        auto new_cell = torch::zeros_like(old_cell);
        auto input_gate = torch::zeros_like(old_cell);
        auto output_gate = torch::zeros_like(old_cell);
        auto candidate_cell = torch::zeros_like(old_cell);

        const int threads = 1024;
        const dim3 blocks((state_size + threads - 1) / threads, batch_size);

        AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
            lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
        }));

        return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
    }


    template <typename scalar_t>
    __global__ void lltm_cuda_backward_kernel(
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
        torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_gates,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell,
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gate_weights) {
        //batch index
        const int n = blockIdx.y;
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < d_gates.size(2)){
            const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
            const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
            const auto d_new_cell =
                d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


            d_old_cell[n][c] = d_new_cell;
            const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
            const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

            d_gates[n][0][c] =
                d_input_gate * d_sigmoid(gate_weights[n][0][c]);
            d_gates[n][1][c] =
                d_output_gate * d_sigmoid(gate_weights[n][1][c]);
            d_gates[n][2][c] =
                d_candidate_cell * d_elu(gate_weights[n][2][c]);
        }
        }

        std::vector<torch::Tensor> lltm_cuda_backward(
            torch::Tensor grad_h,
            torch::Tensor grad_cell,
            torch::Tensor new_cell,
            torch::Tensor input_gate,
            torch::Tensor output_gate,
            torch::Tensor candidate_cell,
            torch::Tensor X,
            torch::Tensor gates,
            torch::Tensor weights) {
        auto d_old_cell = torch::zeros_like(new_cell);
        auto d_gates = torch::zeros_like(gates);

        const auto batch_size = new_cell.size(0);
        const auto state_size = new_cell.size(1);

        const int threads = 1024;
        const dim3 blocks((state_size + threads - 1) / threads, batch_size);

        AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_cuda", ([&] {
            lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                d_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));

        auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
        auto d_weights = d_gate_weights.t().mm(X);
        auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

        auto d_X = d_gate_weights.mm(weights);
        auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
        auto d_input = d_X.slice(/*dim=*/1, state_size);

        return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
    }

使用示例

.. code-block:: cpp

    // libtorch_test.cpp
    #include "lltm_cuda.h"
    #include <iostream>
    #include <torch/torch.h>

    int main() {
        std::cout << torch::cuda::is_available() << std::endl;
        std::cout << torch::cuda::cudnn_is_available() << std::endl;
        std::cout << torch::cuda::device_count() << std::endl;
        // 使用arange构造一个一维向量，再用reshape变换到5x5的矩阵
        torch::Tensor foo = torch::arange(25).reshape({5, 5}).to(torch::kCUDA);

        // 计算矩阵的迹
        torch::Tensor bar = torch::einsum("ii", foo);

        // 输出矩阵和对应的迹
        std::cout << "==> matrix is:\n " << foo << std::endl;
        std::cout << "==> trace of it is:\n " << bar << std::endl;

        int batch_size = 16;
        int input_features = 32;
        int state_size = 128;

        torch::Tensor X = torch::randn({batch_size, input_features}).to(torch::kCUDA);
        torch::Tensor h = torch::randn({batch_size, state_size}).to(torch::kCUDA);
        torch::Tensor C = torch::randn({batch_size, state_size}).to(torch::kCUDA);
        torch::Tensor weights = torch::randn({3 * state_size, input_features + state_size}).to(torch::kCUDA);
        torch::Tensor bias = torch::randn({3 * state_size}).to(torch::kCUDA);
        auto outputs = lltm_forward(X,weights,bias,h,C);
        std::cout<< outputs[0]<<std::endl;
    }

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
    project(libtorchDemo)

    set(CMAKE_CXX_STANDARD 17)


    set(Torch_DIR /home/zack/cpplib/code/libtorch_test/libtorch/share/cmake/Torch)

    find_package(CUDA REQUIRED)
    find_package(Torch REQUIRED)

    include_directories(
        /usr/local/cuda/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        libtorch/include
        libtorch/include/torch/csrc/api/include
    )


    cuda_add_executable(libtorch_test libtorch_test.cpp lltm_cuda.cpp lltm_cuda_kernel.cu)
    target_link_libraries(libtorch_test "${TORCH_LIBRARIES}")
