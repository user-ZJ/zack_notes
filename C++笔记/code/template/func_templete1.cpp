#include <iostream>

// 函数模板，接收lambda、函数对象和函数指针作为参数
template<typename F>
void process(F func)
{
    // 调用传入的参数
    func();
}

// 示例函数对象
struct Functor
{
    void operator()()
    {
        std::cout << "This is a functor" << std::endl;
    }
};

// 示例函数
void function()
{
    std::cout << "This is a function" << std::endl;
}

int main()
{
    // 使用lambda作为参数进行调用
    process([]() {
        std::cout << "This is a lambda" << std::endl;
    });

    // 使用函数对象作为参数进行调用
    Functor functor;
    process(functor);

    // 使用函数指针作为参数进行调用
    process(function);

    return 0;
}