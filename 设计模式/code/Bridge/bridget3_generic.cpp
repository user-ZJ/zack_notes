#include <iostream>
#include <memory>

using namespace std;




//平台实现: n
class PCMessager {
public:
    
    void PlaySound() {
       
       cout<<"PC Play Sound"<<endl;
    }
    void DrawShape() {
        cout<<"PC Draw Shape"<<endl;
    }
    void WriteText() {
        cout<<"PC Write Text"<<endl;
    }
    void Connect() {
        cout<<"PC Connect"<<endl;
    }
};

class MobileMessager {
public:
    
     void PlaySound() {
       
       cout<<"Mobile Play Sound"<<endl;
    }
    void DrawShape() {
        cout<<"Mobile Draw Shape"<<endl;
    }
    void WriteText() {
        cout<<"Mobile Write Text"<<endl;
    }
    void Connect() {
        cout<<"Mobile Connect"<<endl;
    }
};



//业务抽象:m 
template<typename MessagerImp>
class MessagerLite : private  MessagerImp{

public:

    
    void Login(const string& username) {
        
        cout<<"Lite Login"<<endl;
        MessagerImp::Connect();

    }
    void SendMessage(const string& message) {
        
        cout<<"Lite Send Message"<<endl;
        MessagerImp::WriteText();
       
    }
    void SendPicture(const string& path) {
        
        cout<<"Lite Send Picture"<<endl;
        MessagerImp::DrawShape();

    }
};



template<typename MessagerImp>
class MessagerPerfect : private  MessagerImp{

public:
    
    void Login(const string& username) {
        
        
        MessagerImp::Connect();
        MessagerImp::PlaySound();
        cout<<"Perfect  Login"<<endl;
    }
    void SendMessage(const string& message) {
        
        
        //********
        MessagerImp::WriteText();
        MessagerImp::PlaySound();
        cout<<"Perfect Send Message"<<endl;
    }
    void SendPicture(const string& path) {
        

        //********
        MessagerImp::DrawShape();
        MessagerImp::PlaySound();
        cout<<"Perfect Send Picture"<<endl;
    }
};





int main(){
        //运行时装配

        MessagerPerfect<MobileMessager> msg;

       
        msg.Login("jason"s);
        cout<<"---"<<endl;
        msg.SendMessage("hello"s);
        cout<<"---"<<endl;
        msg.SendPicture("/smile.jpg");


}