class ISubject{
public:
    virtual void process()=0;
};



//server-side object
class RealSubject: public ISubject{
public:
    void process() override
    {

    }
};

//**************************************

//client-side proxy
//Proxy的设计
class SubjectProxy: public ISubject{

    RealSubject* realSubject;
public:
    void process() override{
        
        //对RealSubject的一种间接访问
        //....

        //安全控制....
        
        realSubject->process();

    }
};





class ClientApp{
    
    ISubject* subject;
    
public:
    
    ClientApp(){
        subject=new SubjectProxy(...);
    }
    
    void DoTask(){
        //...
        subject->process();
        
        //....
    }
};