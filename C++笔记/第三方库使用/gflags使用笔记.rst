gflags使用笔记
===============

安装
--------------
.. code-block: shell

    mkdir build 
    cd build
    # cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_C_FLAGS="-fPIC" ..
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
    make 
    sudo make install
