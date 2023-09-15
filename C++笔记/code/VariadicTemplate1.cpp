#include <iostream>
#include <string>
#include <sstream>
#include <vector>

template<typename T>
std::string to_string(T value)
{
    std::ostringstream os;
    os << value;
    return os.str();
}

template<typename... Args>
void print(Args... args)
{
    std::vector<std::string> vec{ to_string(args)... }; //递归展开
    for (const auto& str : vec)
    {
        std::cout << str << " ";
    }
    std::cout << std::endl;
}

int main()
{
    print(1, 2, 3, 4, 5);
    print(3.14, 2.718);
    print("hello", "world", "!");
    return 0;
}