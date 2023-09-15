
#include <cstdlib>
#include <cassert>
#include <utility>
#include <iostream>



template <typename T>
class SmartPtr { 

public:
    explicit SmartPtr(T* p = nullptr) 
            p_(p)
    {}
    
    SmartPtr(SmartPtr&& rhs) 
    {
        p_=rhs.p_;
        rhs.p_=nullptr;
    }

    SmartPtr& operator=(SmartPtr&& rhs) 
    {
        delete p_;
        p_=rhs.p_;
        rhs.p_=nullptr;

        return *this;
    }


    ~SmartPtr() { 
        delete p_;
    }
    T* release() { 
        T* temp=p_;
        p_ = NULL; 
        return temp    
    }

    T* get(){
        return p_;
    }

    T* operator->() { return p_; }

    const T* operator->() const { return p_; }

    T& operator*() { return *p_; }

    const T& operator*() const { return *p_; }



private:
    T* p_;
    SmartPtr(const SmartPtr&) = delete;
    SmartPtr& operator=(const SmartPtr&) = delete;
};




int main() {
    
    SmartPtr<int> p(new int(42));
    std::cout << *p << std::endl;

    
}
