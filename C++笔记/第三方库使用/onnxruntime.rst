onnxruntime
===============

安装
-----------------

`参考 <https://onnxruntime.ai/docs/build/inferencing.html>`_

需要安装cmake 3.18以上

.. code-block:: shell

    git clone --recursive https://github.com/Microsoft/onnxruntime
    cd onnxruntime
    ./build.sh --config RelWithDebInfo --build_shared_lib --parallel  #编译debug版本
    ./build.sh --config Release --build_shared_lib --parallel  #编译release版本

    source /opt/intel/openvino_2021/bin/setupvars.sh
    ./build.sh --config RelWithDebInfo --use_openvino CPU_FP32 --build_shared_lib --parallel #编译openvino后端版本，需要先安装好openvino


使用示例
-------------------
.. code-block:: cpp

    #include "onnxruntime_cxx_api.h" 

    // Ort::Env必须要全局唯一，使用单例模式
    class ONNXENV {
      private:
        ONNXENV(){};
        ~ONNXENV(){};
        ONNXENV(const ONNXENV &);
        ONNXENV &operator=(const ONNXENV &);

      public:
        static Ort::Env &getInstance() {
            static Ort::Env env(/*envOpts,*/ ORT_LOGGING_LEVEL_WARNING, "onnx");
            return env;
        }
    };

    Ort::Env &env = ONNXENV::getInstance();
    Ort::SessionOptions session_options;
    // 使用GPU作为后端
    // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    // 或
    //OrtCUDAProviderOptions cuda_options;
	//cuda_options.device_id = 0;
	//cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	//cuda_options.gpu_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
	//cuda_options.arena_extend_strategy = 1;
	//cuda_options.do_copy_in_default_stream = 1;
	//cuda_options.has_user_compute_stream = 1;
	//cuda_options.default_memory_arena_cfg = nullptr;
	session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), session_options);
    // session = std::make_unique<Ort::Session>(env, modelBuff.data(),modelBuff.size(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session->GetInputCount();
    for (int i = 0; i < num_input_nodes; i++) {
        char *input_name = session->GetInputName(i, allocator);
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
    }

    size_t num_output_nodes = session->GetOutputCount();
    for (int i = 0; i < num_output_nodes; i++) {
        char *output_name = session->GetOutputName(i, allocator);
        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
    }

    auto memory_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(input.data()), input.size(),
        input_node_dims_tmp[0].data(), input_node_dims_tmp[0].size()));

    auto output_tensors =
    session->Run(Ort::RunOptions(), input_node_names.data(),
                 ort_inputs.data(), 1/*输入tensor个数*/, output_node_names.data(), 1/*输出tensor个数*/);

    for(const auto &out:output_tensors){
        auto tensor_info = out.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        LOG(INFO)<<printCollection(node_dims);
    }

线程池设置
-------------------------
| SetIntraOpNumThreads: 设置单个算子内部并行计算的线程池大小
| SetInterOpNumThreads: 设置计算图中不同节点并行计算的线程池大小

| SetGlobalIntraOpNumThreads:设置全局线程池大小,创建env时设置，所有session共享，
  如果session中单独使用SetIntraOpNumThreads设置线程池大小，优先使用session中设置
| SetGlobalInterOpNumThreads:同上

https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/global_thread_pools/test_main.cc

.. code-block:: cpp

    const int thread_pool_size = std::thread::hardware_concurrency();
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    std::unique_ptr<OrtStatus, decltype(OrtApi::ReleaseStatus)> st_ptr(nullptr, g_ort->ReleaseStatus);
    OrtThreadingOptions* tp_options;
    st_ptr.reset(g_ort->CreateThreadingOptions(&tp_options));
    st_ptr.reset(g_ort->SetGlobalIntraOpNumThreads(tp_options, thread_pool_size));
    st_ptr.reset(g_ort->SetGlobalInterOpNumThreads(tp_options, thread_pool_size));
    static Ort::Env env(tp_options, ORT_LOGGING_LEVEL_WARNING, "onnxruntime");
    g_ort->ReleaseThreadingOptions(tp_options);

    // session 设置
    session_options.DisablePerSessionThreads();
    // 控制算子之间是否并行
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL;


获取所有层参数
--------------------------------
.. code-block:: cpp

    #include "onnx/onnx_pb.h"
    #include "onnx/onnx-operators_pb.h"
    #include "onnx/onnx.pb.h"
    #include "onnx/onnx-ml.pb.h"

    #include <iostream>
    #include <fstream>

    using namespace ONNX_NAMESPACE;

    void printTensor(const TensorProto& tensor) {
        std::cout << "Name: " << tensor.name() << std::endl;
        std::cout << "Shape: ";
        for (auto dim : tensor.dims()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Data type: " << tensor.data_type() << std::endl;
        std::cout << "Values: ";
        for (auto value : tensor.float_data()) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    void printLayerParams(const NodeProto& node) {
        std::cout << "Layer name: " << node.name() << std::endl;
        std::cout << "Inputs: ";
        for (auto input : node.input()) {
            std::cout << input << " ";
        }
        std::cout << std::endl;
        std::cout << "Outputs: ";
        for (auto output : node.output()) {
            std::cout << output << " ";
        }
        std::cout << std::endl;
        for (auto attr : node.attribute()) {
            std::cout << "Attribute name: " << attr.name() << std::endl;
            if (attr.has_f()) {
                std::cout << "Attribute value: " << attr.f() << std::endl;
            } else if (attr.has_i()) {
                std::cout << "Attribute value: " << attr.i() << std::endl;
            } else if (attr.has_s()) {
                std::cout << "Attribute value: " << attr.s() << std::endl;
            } else if (attr.has_t()) {
                printTensor(attr.t());
            } else if (attr.floats_size() > 0) {
                std::cout << "Attribute values: ";
                for (auto value : attr.floats()) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            } else if (attr.ints_size() > 0) {
                std::cout << "Attribute values: ";
                for (auto value : attr.ints()) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            } else if (attr.strings_size() > 0) {
                std::cout << "Attribute values: ";
                for (auto value : attr.strings()) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            } else if (attr.tensors_size() > 0) {
                std::cout << "Attribute values: ";
                for (auto value : attr.tensors()) {
                    printTensor(value);
                }
                std::cout << std::endl;
            }
        }
    }

    int main() {
        // Load the ONNX model
        ModelProto model;
        std::ifstream ifs("model.onnx", std::ios::binary);
        ifs.seekg(0, std::ios::end);
        int length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        char* buffer = new char[length];
        ifs.read(buffer, length);
        ifs.close();
        model.ParseFromArray(buffer, length);
        delete[] buffer;
        // Traverse the graph and print the layer parameters
        for (auto node : model.graph().node()) {
            printLayerParams(node);
        }
        return 0;
    }

cuda使用
-------------------------------
.. code-block:: cpp

    Ort::SessionOptions session_options;
    // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.gpu_mem_limit = static_cast<int>(4 * 1024 * 1024);
    cuda_options.arena_extend_strategy = 1;
    cuda_options.do_copy_in_default_stream = 1;
    cuda_options.has_user_compute_stream = 1;
    cuda_options.default_memory_arena_cfg = nullptr;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

openvino使用
--------------------------

https://github.com/yas-sim/openvino-ep-enabled-onnxruntime


示例：
https://gitee.com/mirrors_microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc
