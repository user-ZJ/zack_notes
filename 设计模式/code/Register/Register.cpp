#include <iostream>
#include <unordered_map>
#include <string>

// 可注册对象接口
class Registrable {
public:
    virtual ~Registrable() {}
    virtual void DoSomething() = 0;
};

// 具体可注册对象
class ConcreteRegistrable : public Registrable {
public:
    void DoSomething() {
        std::cout << "Doing something..." << std::endl;
    }
};

// 注册器类
class Registry {
private:
    std::unordered_map<std::string, Registrable*> objects;

public:
    void Register(const std::string& name, Registrable* obj) {
        objects[name] = obj;
    }

    Registrable* GetObject(const std::string& name) {
        auto it = objects.find(name);
        return it != objects.end() ? it->second : nullptr;
    }
};

int main() {
    Registry registry;

    // 创建可注册对象并注册到注册器中
    ConcreteRegistrable obj1;
    registry.Register("obj1", &obj1);

    // 从注册器中获取已注册的对象并使用
    Registrable* obj = registry.GetObject("obj1");
    if (obj) {
        obj->DoSomething();
    }

    return 0;
}