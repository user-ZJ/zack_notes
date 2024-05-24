C++调用python
=====================================

使用python提供给C/C++的API
----------------------------------------
主流方法将python程序编程文本形式的动态链接库，在c/c++程序中调用其中定义的函数。

本质上是在 c++ 中启动了一个 python 解释器，由解释器对 python 相关的代码进行执行，执行完毕后释放资源，达到调用目的

第一个示例
---------------------
.. code-block:: cpp

    #include "python/Python.h"
    #include <iostream>

    int main()
    {
        Py_Initialize();    // 初始化
        if(!Py_IsInitialized()){
            std::cout << "python init fail" << std::endl;
            return 0;
        }
        PyRun_SimpleString("print 'hello'");
        Py_Finalize();      // 释放资源
    }

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    project(CppCallPython)
    # 查找Python解释器
    include_directories("/usr/include/python3.8")
    # 添加可执行文件
    add_executable(CppCallPython main.cpp)
    # 链接Python库
    target_link_libraries(CppCallPython python3.8)


输入string,输出string函数调用
-----------------------------------------------------
C++调用python时，可以将输入转换为json字符串，输出也转化为json字符串进行调用

.. code-block:: cpp

    #include <Python.h>
    #include <iostream>
    int main() {
        Py_Initialize();
        // 检查初始化是否成功
        if (!Py_IsInitialized()) {
            std::cout << "python init fail" << std::endl;
            return -1;
        }
        PyRun_SimpleString("print('hello')");
        // 初始化python系统文件路径，保证可以访问到 .py文件
        PyRun_SimpleString("import sys");
	    PyRun_SimpleString("sys.path.append('./script')");
        // 调用python文件名，不用写后缀
        PyObject* pModule = PyImport_ImportModule("example");
        if( pModule == NULL ){
            std::cout <<"module not found" << std::endl;
            PyErr_Print();
            return 1;
        }
        // 调用函数
        PyObject* pFunc = PyObject_GetAttrString(pModule, "greet");
        if( !pFunc || !PyCallable_Check(pFunc)){
            std::cout <<"not found function add_num" << std::endl;
            return 0;
        }
        std::string input = "my is c++ test!";
        auto pArg = Py_BuildValue("(s)", input.c_str());
        // 如果没有参数，pArg传入NULL
        auto pValue = PyEval_CallObject(pFunc, pArg); 
        // 获取返回值
        PyObject* pValueStr = PyObject_Str(pValue);
        const char* pValueChars = PyUnicode_AsUTF8(pValueStr);
        std::string result(pValueChars);
        std::cout << result << std::endl; 
        Py_DECREF(pArg);
        Py_Finalize();
        return 0;
    }

输入bytes,输出string函数调用
-----------------------------------------------------
C++调用python时，文件输入，输出为json字符串进行调用

.. code-block:: cpp

    #include <Python.h>
    #include <iostream>
    int main() {
        Py_Initialize();
        // 检查初始化是否成功
        if (!Py_IsInitialized()) {
            std::cout << "python init fail" << std::endl;
            return -1;
        }
        PyRun_SimpleString("print('hello')");
        // 初始化python系统文件路径，保证可以访问到 .py文件
        PyRun_SimpleString("import sys");
	    PyRun_SimpleString("sys.path.append('./script')");
        // 调用python文件名，不用写后缀
        PyObject* pModule = PyImport_ImportModule("example");
        if( pModule == NULL ){
            std::cout <<"module not found" << std::endl;
            PyErr_Print();
            return 1;
        }
        // 调用函数
        PyObject* pFunc = PyObject_GetAttrString(pModule, "greet");
        if( !pFunc || !PyCallable_Check(pFunc)){
            std::cout <<"not found function add_num" << std::endl;
            return 0;
        }
        //创建一个元组来存放参数
        PyObject* pArgs = PyTuple_New(1);
        PyObject* pBytes  = PyBytes_FromStringAndSize(buff.data(),buff.size());
        //将bytes对象添加到元组中
        PyTuple_SetItem(pArgs, 0, pBytes);
        // 如果没有参数，pArg传入NULL
        auto pValue = PyEval_CallObject(pFunc, pArgs); 
        // 获取返回值
        PyObject* pValueStr = PyObject_Str(pValue);
        const char* pValueChars = PyUnicode_AsUTF8(pValueStr);
        std::string result(pValueChars);
        std::cout << result << std::endl; 
        Py_DECREF(pArgs);
        Py_Finalize();
        return 0;
    }


使用单例初始化
-----------------------------------------------------
使用Py_Finalize释放资源后，再使用Py_Initialize后，二次导入numpy、docx等包会报错，
需要使用单例模式进行封装，在构造函数中Py_Initialize，在析构函数中Py_Finalize

.. code-block:: cpp

    class PythonCall{
      public:
        ~PythonCall();
        std::string getDocxText(const std::string &buff);
        std::string getDocxParse(const std::string &buff);

        static PythonCall &instance(){
            static PythonCall ins;
            return ins;
        }

      private:
        PythonCall();
    };

    PythonCall::PythonCall(){
        Py_Initialize();
        // 检查初始化是否成功
        if (!Py_IsInitialized()) {
            LOG(ERROR) << "python init fail";
        }
    }

    PythonCall::~PythonCall(){
        Py_Finalize();
    }


参考
---------------
https://zhuanlan.zhihu.com/p/450318119

https://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I