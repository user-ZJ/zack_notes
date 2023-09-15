mlc-llm环境配置
================================

安装TVM Unity
-----------------------
.. note:: 

    要求cuda版本大于11.8

.. code-block:: shell

    apt install llvm-dev
    git clone --recursive https://github.com/mlc-ai/relax.git tvm-unity
    cd tvm-unity
    rm -rf build && mkdir build && cd build
    cp ../cmake/config.cmake .
    # controls default compilation flags
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
    # LLVM is a must dependency
    echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
    # GPU SDKs, turn on if needed
    echo "set(USE_CUDA   ON)" >> config.cmake
    echo "set(USE_METAL  OFF)" >> config.cmake
    echo "set(USE_VULKAN OFF)" >> config.cmake
    echo "set(USE_OPENCL OFF)" >> config.cmake
    cmake .. && cmake --build . --parallel $(nproc)
    cd /path-to-tvm-unity/python
    pip install -e .

验证tvm安装
`````````````````````````
.. code-block:: shell

    # 查看tvm位置
    python -c "import tvm; print(tvm.__file__)"
    # 查看tvm使用的库
    python -c "import tvm; print(tvm._ffi.base._LIB)"
    # 查看tvm编译选项
    python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"

    # 检测设备支持
    python -c "import tvm; print(tvm.metal().exist)"
    python -c "import tvm; print(tvm.cuda().exist)"
    python -c "import tvm; print(tvm.vulkan().exist)"


安装mlc-llm
---------------------------
.. code-block:: shell

    apt install cargo
    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm
    mkdir -p build && cd build
    python3 ../cmake/gen_cmake_config.py
    cmake .. && cmake --build . --parallel $(nproc) && cd ..
    # 验证安装
    ./build/mlc_chat_cli --help
    # 安装python包
    cd mlc-llm
    pip install -e .
    python -c "import mlc_llm; print(mlc_llm)"
    python3 -m mlc_llm.build --help

编译模型
----------------------------
.. code-block:: shell

    python3 -m mlc_llm.build --model Llama-2-7b-chat-hf --target cuda --quantization q4f16_1
    # Run CLI
    mlc_chat_cli --model Llama-2-7b-chat-hf-q4f16_1
