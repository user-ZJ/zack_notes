ggml使用笔记
==========================
ggml是一个Tensor运算库，可以用来实现线性回归，支持向量机，神经网络

该库允许用户使用可用的张量操作来定义某个函数。这个函数内部通过计算图表示。函数定义中的每个张量操作对应于图中的一个节点。
定义了计算图后，用户可以选择计算函数的值和/或其相对于输入变量的梯度。可以选择使用一种可用的优化算法对函数进行优化。

示例
-------------
.. code-block:: cpp

    //f(x) = a*x^2 + b
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };

    // memory allocation happens here
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_param(ctx, x); // x is an input variable

    struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
    struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);

    struct ggml_cgraph gf = ggml_build_forward(f);
    // set the input variable and parameter values
    ggml_set_f32(x, 2.0f);
    ggml_set_f32(a, 3.0f);
    ggml_set_f32(b, 4.0f);
    // 实际计算
    ggml_graph_compute_with_ctx(ctx, &gf, n_threads);

    printf("f = %f\n", ggml_get_f32_1d(f, 0));


数据结构
----------------
* ``ggml_tensor`` :描述tensor的size,datatype,使用的是什么地方的内存，指向输入tensor的指针列表。多维tensor默认是以行优先顺序存储的

.. code-block:: cpp

    enum ggml_backend {
        GGML_BACKEND_CPU = 0,
        GGML_BACKEND_GPU = 10,
        GGML_BACKEND_GPU_SPLIT = 20,
    };

    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type    type;
        enum ggml_backend backend;
        int     n_dims;
        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = sizeof(type)
                                   // nb[1] = nb[0]   * ne[0] + padding
                                   // nb[i] = nb[i-1] * ne[i-1]
        // compute data
        enum ggml_op op;
        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
        bool is_param;
        struct ggml_tensor * grad;
        struct ggml_tensor * src[GGML_MAX_SRC];
        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;
        char name[GGML_MAX_NAME];
        void * extra; // extra things e.g. for ggml-cuda.cu
        char padding[4];
    };

.. code-block:: cpp

    struct ggml_context {
        size_t mem_size;
        void * mem_buffer;
        bool   mem_buffer_owned;
        bool   no_alloc;
        // this is used to save the no_alloc state when using scratch buffers
        bool   no_alloc_save; 
        int    n_objects;
        struct ggml_object * objects_begin;
        struct ggml_object * objects_end;
        // scratch 临时内存
        struct ggml_scratch scratch;
        struct ggml_scratch scratch_save;
    };


* ``struct ggml_context * ggml_init(struct ggml_init_params params);`` 创建ggml上下文，上下文用于管理ggml内存
* ``size_t  ggml_used_mem(const struct ggml_context * ctx);``  实际使用内存量
* ``struct ggml_tensor * ggml_new_tensor(struct ggml_context *ctx,enum ggml_type type,int n_dims,const int64_t *ne);`` 使用ggml_init申请的内存创建Tensor
* ``struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context *ctx,enum ggml_type type,int64_t ne0);`` 使用ggml_init申请的内存创建1维的Tensor
* ``struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context *ctx,enum ggml_type type,int64_t ne0,int64_t ne1);`` 创建2维Tensor
* ``struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context *ctx,enum ggml_type type,int64_t ne0,int64_t ne1,int64_t ne2);`` 创建3维Tensor
* ``struct ggml_tensor * ggml_new_tensor_4d(struct ggml_context *ctx,enum ggml_type type,int64_t ne0,int64_t ne1,int64_t ne2,int64_t ne3);`` 创建4维Tensor
* ``struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);`` 
* ``struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);``
* ``struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);``  复制Tensor
* ``struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);`` 
* ``struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);``  将Tensor赋值为0
* ``struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);``
* ``struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);``
* ``int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);``
* ``void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);``
* ``float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);``
* ``void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);``
* ``void *  ggml_get_data(const struct ggml_tensor * tensor);``
* ``float * ggml_get_data_f32(const struct ggml_tensor * tensor);``

* ``ggml_mul``
* ``ggml_add``
* ``ggml_permute``
* ``ggml_conv_1d``

* ``struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);``  创建一个计算图来计算Tensor
* ``void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);`` 将tensor加入图中计算
* ``ggml_set_param``   将tensor设置为输入变量
* ``ggml_graph_compute_with_ctx``






参考
-----------------
https://github.com/ggerganov/ggml

https://github.com/ggerganov/llama.cpp