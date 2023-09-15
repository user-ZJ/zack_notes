//-std=c++20 -lstdc++ -fmodules-ts -c
module;

#include <iostream>

// 声明模块MyModule
export module MyModule; 


struct MyData{
    int value;
};


//模块对外接口
export class MyClass{
    MyData data;
public:
    MyClass(int value):data{value}{

    }

    MyData getData() const{
        return data;
    }

    
};



 




