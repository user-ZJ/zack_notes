#include <iostream>
#include <string>
#include <mutex>

using namespace std;
class Singleton {
public:
    // 获取Singleton实例的静态方法
    static Singleton* getInstance() {
        // 使用双检查锁确保在多线程环境下仅创建一次实例
         if (instance == nullptr) {
            std::lock_guard<std::mutex> lock(mutex);
            if (instance == nullptr) {
                instance = new Singleton();
            }
        }
        return instance;
    }

    // 禁止拷贝构造函数和拷贝赋值操作符
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

private:
    // 将构造函数和析构函数声明为私有，以防止外部创建多个实例或者销毁实例
    Singleton() {}
    ~Singleton() {}

    // 声明静态变量和互斥量
    static Singleton* instance;
    static std::mutex mutex;
};

// 初始化静态变量
Singleton* Singleton::instance = nullptr;
std::mutex Singleton::mutex;

int main(){

    Singleton* s1=Singleton::getInstance();
    Singleton* s2=Singleton::getInstance();

    cout<<s1<<endl;
    cout<<s2<<endl;

}