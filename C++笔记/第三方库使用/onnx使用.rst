onnx使用
==================

1. git clone https://github.com/onnx/onnx.git
2. cd onnx
3. mkdir build
4. cd build
5. cmake -DCMAKE_INSTALL_PREFIX=../install ..
6. make -j 4
7. make install 

onnx使用实例
-------------------------------

CMakeLists.txt
```````````````````````````
.. code-block:: cmake

    cmake_minimum_required(VERSION 3.14)
    project(test)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

    include_directories(
        install/include
            )
    link_directories(
        install/lib
            )
    set(CMAKE_PREFIX_PATH install/lib/cmake)
  
    find_package(protobuf REQUIRED)
    find_package(ONNX REQUIRED)

    add_definitions(-DONNX_ML=1)
    add_definitions(-DONNX_NAMESPACE=onnx)

    add_executable(test_onnx test_onnx.cpp)
    target_link_libraries(test_onnx onnx_proto)

test_onnx.cpp
`````````````````````````
.. code-block:: cpp

    #include "onnx/onnx_pb.h"
    #include "onnx/proto_utils.h"
    #include <fstream>
    #include <iostream>

    int main(void) {
        onnx::ModelProto model;
        std::ifstream in("mnist.onnx", std::ios_base::binary);
        model.ParseFromIstream(&in);
        // onnx::ParseProtoFromBytes(&model, bytes, nbytes);
        in.close();
        std::cout << "input size:" << model.graph().input().size() << "\n";
        std::cout << "input initializer:" << model.graph().initializer().size() << "\n";
        auto &initializer = model.graph().initializer();
        for (int i = 0; i < initializer.size(); i++) {
            std::cout << "initializer name:" << initializer[i].name() << " data_type:" << initializer[i].data_type() << " "
                    << initializer[i].DataType_Name(initializer[i].data_type()) << "\n";
            auto &dims = initializer[i].dims();
            for (int j = 0; j < dims.size(); j++)
            std::cout << dims[j] << ",";
            std::cout << "\n";
            std::cout<<initializer[i].has_raw_data()<<" "<<initializer[i].raw_data().size() <<"\n";
        }
    }



apt-get install -y protobuf-compiler libprotoc-dev