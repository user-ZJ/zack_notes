#include <iostream>

template<typename... Args>
void print(Args... args) {
    ((std::cout << args << " "), ...); // 折叠表达式
    std::cout<<"\n";
}

int main() {
    print(1, 2.0, "three", '4');
    return 0;
}