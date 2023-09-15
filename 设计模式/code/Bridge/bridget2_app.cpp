#include <iostream>
#include <memory>

using namespace std;


//实现
class MessagerImp {
public:
    virtual void PlaySound()=0;
    virtual void DrawShape()=0;
    virtual void WriteText()=0;
    virtual void Connect()=0;
    
    virtual ~MessagerImp(){}
};


//抽象
class Messager{
protected:
    unique_ptr<MessagerImp> pmb;//=new PCMessager(); MobileMessager();
public:

    Messager(unique_ptr<MessagerImp> _pmb):pmb(std::move(_pmb))
    {

    }

    virtual void Login(const string& username)=0;
    virtual void SendMessage(const string& message)=0;
    virtual void SendPicture(const string& path)=0;

    virtual ~Messager(){}
};


// 1+1+n+m =12


//平台实现: n
class PCMessager : public MessagerImp{
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

class MobileMessager : public MessagerImp{
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



//业务抽象:m 
class MessagerLite : public  Messager{

public:

    MessagerLite(unique_ptr<MessagerImp> _pmb):Messager(std::move(_pmb))
    {

    }
    
    void Login(const string& username) override{
        
        cout<<"Lite Login"<<endl;
        pmb->Connect();

    }
    void SendMessage(const string& message) override{
        
        cout<<"Lite Send Message"<<endl;
        pmb->WriteText();
       
    }
    void SendPicture(const string& path) override{
        
        cout<<"Lite Send Picture"<<endl;
        pmb->DrawShape();

    }
};



class MessagerPerfect :public  Messager {
 
public:

    MessagerPerfect(unique_ptr<MessagerImp> _pmb):Messager(std::move(_pmb))
    {

    }
    
    
    void Login(const string& username) override{
        
        
        pmb->Connect();
        pmb->PlaySound();
        cout<<"Perfect  Login"<<endl;
    }
    void SendMessage(const string& message) override{
        
        
        //********
        pmb->WriteText();
        pmb->PlaySound();
        cout<<"Perfect Send Message"<<endl;
    }
    void SendPicture(const string& path) override{
        

        //********
        pmb->DrawShape();
        pmb->PlaySound();
        cout<<"Perfect Send Picture"<<endl;
    }
};





int main(){
        //运行时装配

        unique_ptr<Messager> pMsg= 
            std::make_unique<MessagerPerfect>(make_unique<MobileMessager>());

       
        pMsg->Login("jason"s);
        cout<<"---"<<endl;
        pMsg->SendMessage("hello"s);
        cout<<"---"<<endl;
        pMsg->SendPicture("/smile.jpg");


}