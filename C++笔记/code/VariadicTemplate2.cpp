#include <iostream>
#include <tuple>

template<typename... Args>
class Tuple {};

template<typename T, typename... Args>
class Tuple<T, Args...> : public Tuple<Args...> {
public:
    Tuple(T value, Args... args) : Tuple<Args...>(args...), value(value) {}

    T& get() {
        return value;
    }

private:
    T value;
};

template<typename T>
class Tuple<T> {
public:
    Tuple(T value) : value(value) {}

    T& get() {
        return value;
    }

private:
    T value;
};

int main() {
    Tuple<int, double, std::string> myTuple(42, 3.14, "hello world");
    std::cout << myTuple.get() << std::endl;
    std::cout << myTuple.Tuple<double, std::string>::get() << std::endl;
    std::cout << myTuple.Tuple<std::string>::get() << std::endl;
    return 0;
}