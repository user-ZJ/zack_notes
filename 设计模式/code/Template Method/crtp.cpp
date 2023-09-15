#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
using namespace std;

// 基类参数T包含了子类编译时信息
template <typename T> 
class Base {
public:

    Base() : i_(0) { }

    void process() { sub()->process_imp(); }
    void process_imp() { 
      cout<<"Base::process()"<<endl;
    }

    //将基类指针转型为子类T的指针
    T* sub() { return static_cast<T*>(this); }

    int get() const { return i_; }

    ~Base(){
      cout<<"~Base()"<<endl;
      
    }

    void destroy(){
      delete sub();
    }

protected:
    int i_;
};



class Sub : public Base<Sub> {
public:
    int value;
    
    ~Sub()
    {
      cout<<"~Sub"<<endl;
    }

    void process_imp() { 
      cout<<"Sub::process()"<<endl;
    }
};



int main()
{
  
  {
    Base<Sub>* pBase=new Sub();
    pBase->process();

    cout<<pBase<<endl;
    cout<<static_cast<Sub*>(pBase)<<endl;
    
    //delete pBase;
    
    pBase->destroy();
  }

  {
    using SBase=Base<Sub>;
    auto lambda = []( SBase* p) { p->destroy(); };

    unique_ptr<SBase, decltype(lambda)> uptr(new Sub(), lambda);
    uptr->process();
  }

}