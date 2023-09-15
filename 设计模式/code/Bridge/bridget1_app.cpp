#include <iostream>
#include <memory>

using namespace std;


// 1. 支持多平台
// 2. 多业务版本
class Messager{
public:

    //业务抽象责任：变化的方向（抽象）
    virtual void Login(const string& username)=0;
    virtual void SendMessage(const string& message)=0;
    virtual void SendPicture(const string& path)=0;

    //平台实现责任：变化的方向（实现）
    virtual void PlaySound()=0;
    virtual void DrawShape()=0;
    virtual void WriteText()=0;
    virtual void Connect()=0;
    virtual ~Messager(){}
};

// n=5, m=5
// 1 +n + n*m  = 1+5+5*5= 31


//多平台实现: n 
class PCMessagerBase : public Messager{
public:
    
    void PlaySound() override{
       
       cout<<"PC Play Sound"<<endl;
    }
    void DrawShape() override{
        cout<<"PC Draw Shape"<<endl;
    }
    void WriteText() override{
        cout<<"PC Write Text"<<endl;
    }
    void Connect() override{
        cout<<"PC Connect"<<endl;
    }
};

class MobileMessagerBase : public Messager{
public:
    
    void PlaySound() override{
       
       cout<<"Mobile Play Sound"<<endl;
    }
    void DrawShape() override{
        cout<<"Mobile Draw Shape"<<endl;
    }
    void WriteText() override{
        cout<<"Mobile Write Text"<<endl;
    }
    void Connect() override{
        cout<<"Mobile Connect"<<endl;
    }
};


// 多业务版本: m 

class PCMessagerLite : public PCMessagerBase {
public:
    void Login(const string& username) override{
        
        cout<<"Lite Login"<<endl;
        PCMessagerBase::Connect();
        //........
    }
    void SendMessage(const string& message) override{
        
        cout<<"Lite Send Message"<<endl;
        PCMessagerBase::WriteText();
        //........
    }
    void SendPicture(const string& path) override{
        
        cout<<"Lite Send Picture"<<endl;
        PCMessagerBase::DrawShape();
        //........
    }
};


class PCMessagerPerfect : public PCMessagerBase {
public:
    
    void Login(const string& username) override{
        
        PCMessagerBase::Connect();
        PCMessagerBase::PlaySound();
        cout<<"Perfect  Login"<<endl;
        
        //........
    }
    void SendMessage(const string& message) override{
        
        PCMessagerBase::WriteText();
        PCMessagerBase::PlaySound();
        cout<<"Perfect Send Message"<<endl;
        //........
    }
    void SendPicture(const string& path) override{
        
        PCMessagerBase::DrawShape();
        PCMessagerBase::PlaySound();
        cout<<"Perfect Send Picture"<<endl;
        //........
    }
};



class MobileMessagerLite : public MobileMessagerBase {
public:
 void Login(const string& username) override{
        
        cout<<"Lite Login"<<endl;
        MobileMessagerBase::Connect();
        //........
    }
    void SendMessage(const string& message) override{
        
        cout<<"Lite Send Message"<<endl;
        MobileMessagerBase::WriteText();
        //........
    }
    void SendPicture(const string& path) override{
        
        cout<<"Lite Send Picture"<<endl;
        MobileMessagerBase::DrawShape();
        //........
    }

};


class MobileMessagerPerfect : public MobileMessagerBase {
public:
    
      void Login(const string& username) override{
        
        MobileMessagerBase::Connect();
        MobileMessagerBase::PlaySound();
        cout<<"Perfect  Login"<<endl;
        
        //........
    }
    void SendMessage(const string& message) override{
        
        MobileMessagerBase::WriteText();
        MobileMessagerBase::PlaySound();
        cout<<"Perfect Send Message"<<endl;
        //........
    }
    void SendPicture(const string& path) override{
        
        MobileMessagerBase::DrawShape();
        MobileMessagerBase::PlaySound();
        cout<<"Perfect Send Picture"<<endl;
        //........
    }
    
};





int main()
{
    
        auto pMsg =
            make_unique<MobileMessagerPerfect>();

        pMsg->Login("jason"s);
        cout<<"---"<<endl;
        pMsg->SendMessage("hello"s);
        cout<<"---"<<endl;
        pMsg->SendPicture("/smile.jpg");


}