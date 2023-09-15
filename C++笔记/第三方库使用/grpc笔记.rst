======================================
grpc 笔记
======================================

安装
==================

编译源码
------------

.. code-block:: shell

    export MY_INSTALL_DIR=$HOME/.local
    mkdir -p $MY_INSTALL_DIR
    export PATH="$MY_INSTALL_DIR/bin:$PATH"
    # cmake 要求3.13以上
    wget -q -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-Linux-x86_64.sh
    sh cmake-linux.sh -- --skip-license --prefix=$MY_INSTALL_DIR
    # 安装依赖库
    sudo apt install -y build-essential autoconf libtool pkg-config
    # 下载源码
    git clone --recursive -b v1.49.x https://github.com/grpc/grpc.git
    # 编译
    cd grpc
    mkdir -p cmake/build
    pushd cmake/build
    cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR ../.. #-DBUILD_SHARED_LIBS=ON
    make -j 4
    make install
    popd


编译helloworld
----------------------

.. code-block:: shell

    cd examples/cpp/helloworld
    mkdir -p cmake/build
    pushd cmake/build
    cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
    make -j
    # 运行
    ./greeter_server &
    ./greeter_client

编译route_guide
-------------------------------

.. code-block:: shell

    cd examples/cpp/route_guide
    mkdir -p cmake/build
    cd cmake/build
    cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
    make -j 4

示例
--------------------
创建calculator.proto文件
`````````````````````````````````````
.. code-block:: proto

    syntax = "proto3";

    package calculator;

    service Calculator {
        rpc Add(AddRequest) returns (AddResponse) {}
        rpc Multiply(MultiplyRequest) returns (MultiplyResponse) {}
    }

    message AddRequest {
        int32 a = 1;
        int32 b = 2;
    }

    message AddResponse {
        int32 result = 1;
    }

    message MultiplyRequest {
        int32 a = 1;
        int32 b = 2;
    }

    message MultiplyResponse {
        int32 result = 1;
    }

使用protoc编译器和gRPC插件生成C++代码
```````````````````````````````````````````
.. code-block:: shell

    ./protoc --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=./grpc_cpp_plugin calculator.proto 

服务端
`````````````````````````
.. code-block:: cpp

    #include <grpcpp/grpcpp.h>
    #include "calculator.pb.h"
    #include "calculator.grpc.pb.h"


    class CalculatorServiceImpl final : public calculator::Calculator::Service {
        grpc::Status Add(grpc::ServerContext* context, const calculator::AddRequest* request, calculator::AddResponse* response) override {
            response->set_result(request->a() + request->b());
            return grpc::Status::OK;
    }

    grpc::Status Multiply(grpc::ServerContext* context, const calculator::MultiplyRequest* request, calculator::MultiplyResponse* response) override {
        response->set_result(request->a() * request->b());
        return grpc::Status::OK;
    }
    };

    void RunServer() {
        std::string server_address("0.0.0.0:50051");
        CalculatorServiceImpl service;

        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        grpc::EnableDefaultHealthCheckService(true);
        // 线程池
        builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS,4);
        std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
        std::cout << "Server listening on " << server_address << std::endl;

        server->Wait();
    }

    int main(int argc, char** argv) {
        RunServer();
        return 0;
    }

客户端
```````````````````
.. code-block:: cpp

    #include <iostream>
    #include <memory>
    #include <string>
    #include <grpcpp/grpcpp.h>
    #include "calculator.grpc.pb.h"


    class CalculatorClient {
    public:
        CalculatorClient(std::shared_ptr<grpc::Channel> channel)
            : stub_(calculator::Calculator::NewStub(channel)) {}

        int Add(int a, int b) {
            calculator::AddRequest request;
            request.set_a(a);
            request.set_b(b);

            calculator::AddResponse response;
            grpc::ClientContext context;

            grpc::Status status = stub_->Add(&context, request, &response);

            if (status.ok()) {
            return response.result();
            } else {
            std::cout << "RPC failed: " << status.error_message() << std::endl;
            return -1;
            }
        }

        int Multiply(int a, int b) {
            calculator::MultiplyRequest request;
            request.set_a(a);
            request.set_b(b);

            calculator::MultiplyResponse response;
            grpc::ClientContext context;

            grpc::Status status = stub_->Multiply(&context, request, &response);

            if (status.ok()) {
            return response.result();
            } else {
            std::cout << "RPC failed: " << status.error_message() << std::endl;
            return -1;
            }
        }

    private:
        std::unique_ptr<calculator::Calculator::Stub> stub_;
    };

    int main(int argc, char** argv) {
        CalculatorClient calculator(grpc::CreateChannel(
            "localhost:50051", grpc::InsecureChannelCredentials()));
        int a = 10, b = 20;
        int result = calculator.Add(a, b);
        std::cout << a << " + " << b << " = " << result << std::endl;
        result = calculator.Multiply(a, b);
        std::cout << a << " * " << b << " = " << result << std::endl;
        return 0;
    }

cmake
----------------
.. code-block:: cmake

    cmake_minimum_required(VERSION 3.5)
    project(MyProject)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_PREFIX_PATH /data/code/grpc/cmake/install)
    # set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")
    find_package(Protobuf REQUIRED)
    find_package(gRPC REQUIRED)
    set(GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    # set(GRPC_CPP_PLUGIN_EXECUTABLE /data/code/grpc/cmake/install/bin/grpc_cpp_plugin)
    set(GENS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gens)
    set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
    set(PROTO_LIST calculator)
    foreach(_proto_file ${PROTO_LIST})
        get_filename_component(_proto_name ${_proto_file} NAME_WE)
        set(_proto_cpp_file "${GENS_DIR}/${_proto_name}.pb.cc")
        set(_proto_h_file "${GENS_DIR}/${_proto_name}.pb.h")
        set(_grpc_cpp_file "${GENS_DIR}/${_proto_name}.grpc.pb.cc")
        set(_grpc_h_file "${GENS_DIR}/${_proto_name}.grpc.pb.h")
        set(_grpc_file "${PROTO_DIR}/${_proto_file}.proto")
        add_custom_command(
            OUTPUT ${_proto_cpp_file} ${_proto_h_file} ${_grpc_cpp_file} ${_grpc_h_file}
            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
            ARGS --grpc_out ${GENS_DIR}
            --cpp_out ${GENS_DIR}
            -I ${PROTO_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
            ${_grpc_file}
            DEPENDS ${_grpc_file} ${PROTOBUF_PROTOC_EXECUTABLE} ${GRPC_CPP_PLUGIN_EXECUTABLE} 
            COMMENT "Running grpc_cpp_plugin on ${_proto_file}"
            VERBATIM)
        list(APPEND GENS_SRCS ${_grpc_cpp_file})
        list(APPEND GENS_HDRS ${_grpc_h_file})
        list(APPEND GENS_SRCS ${_proto_cpp_file})
        list(APPEND GENS_HDRS ${_proto_h_file})
    endforeach()

    include_directories(${GENS_DIR})

    add_executable(calculator_server calculator_server.cpp ${GENS_SRCS})
    target_link_libraries(calculator_server gRPC::grpc++ gRPC::grpc++_reflection ${PROTOBUF_LIBRARIES})

    add_executable(calculator_client calculator_client.cpp ${GENS_SRCS})
    target_link_libraries(calculator_client gRPC::grpc++ gRPC::grpc++_reflection ${PROTOBUF_LIBRARIES})



参考
=============
https://grpc.io/docs/languages/cpp/quickstart/

