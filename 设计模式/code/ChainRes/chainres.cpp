#include <iostream>
#include <memory>

using namespace std;

class Responder
{
public:
    virtual ~Responder() {}

    void setSuccessor( Responder *s )
    {
        successor = s;
    }

    //沿着链表处理
    void handleAlert()
    {
        if ( this->canHandle() )
        {
            this->handle();
        }
        else
        {
            if (successor != nullptr)
            {
                successor->handleAlert();
            }
            else
            {
                cout<<"NO Response"<<endl; //缺省处理
            }
        }
    }

private:

    virtual bool canHandle()=0;

    virtual void handle()=0;

    Responder *successor {nullptr} ;
  
};


class EmailResponder : public Responder
{
public:
    ~EmailResponder() {}

private:

    bool canHandle() override
    {
        return false;
    }

    void handle() override
    {
        std::cout << "email the alert" << std::endl;  
    }
};

class SMSResponder : public Responder
{
public:
    ~SMSResponder() {}

private:

    bool canHandle()  override
    {
        return false;
    }

    void handle() override
    {
  
        std::cout << "sms the alert" << std::endl;
       
    }
};

class PhoneResponder : public Responder
{
public:
    ~PhoneResponder() {}

private:

    bool canHandle()  override
    {
        return false;
    }

    void handle() override
    {
  
        std::cout << "call to tell the alert" << std::endl;
       
    }
};

class WhistleResponder : public Responder
{
public:
    ~WhistleResponder() {}

private:

    bool canHandle()  override
    {
        return true;
    }

    void handle() override
    {
  
        std::cout << "whistle to tell the alert" << std::endl;
       
    }
};



using ResponderPtr=unique_ptr<Responder>;

int main()
{
    ResponderPtr responder1=make_unique<EmailResponder>() ;
    ResponderPtr responder2=make_unique<SMSResponder>() ;
    ResponderPtr responder3=make_unique<PhoneResponder>() ;
    ResponderPtr responder4=make_unique<WhistleResponder>() ;

    // 1- > 2 -> 3 ->4
    responder1->setSuccessor(responder2.get());
    responder2->setSuccessor(responder3.get());
    responder3->setSuccessor(responder4.get());

    responder1->handleAlert();


    return 0;
}