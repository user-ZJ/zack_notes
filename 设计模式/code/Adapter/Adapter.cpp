


//遗留接口（老接口）
class IAdaptee{
public:
    virtual void invoke(int data)=0;
    virtual int getData()=0;
};


//遗留类型
class OldClass: public IAdaptee{
    //....
};


//目标接口（新接口）
class ITarget{
public:
    virtual void process()=0;
};




//对象适配器
class Adapter: public ITarget{ //继承--接口规约
protected:
    unique_ptr<IAdaptee> pAdaptee;//组合--复用实现
public:
    Adapter(unique_ptr<IAdaptee> p): pAdaptee( std::move(p)) {

   
    }  
    void process() override {

        //老接口--> 新接口
        int data=pAdaptee->getData();
        
        pAdaptee->invoke(data); 
    }
    
    
};


/*
//类适配器--不鼓励
class Adapter: public ITarget,
               private OldClass{ //多继承
               
               
};
*/

int main(){
  
    unique_ptr<ITarget> pTarget=make_unique<Adapter>(make_unique<OldClass>());
    
    pTarget->process();
    
}










