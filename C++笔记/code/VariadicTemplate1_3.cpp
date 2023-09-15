#include <iostream>

// 递归终止条件
void print() {std::cout<<"\n";}

// 递归函数
template<typename T, typename... Args>
void print(T t, Args... args) {
    std::cout << t << " ";
    print(args...);
}

int main() {
    print(1, 2.0, "three", '4');
    return 0;
}