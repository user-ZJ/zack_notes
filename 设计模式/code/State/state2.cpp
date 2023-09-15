class NetworkState{

protected:
    NetworkState* pNext;
public:
   NetworkState* getNextState(){
    return pNext;
   }

    virtual void Operation1()=0;
    virtual void Operation2()=0;
    virtual void Operation3()=0;

    virtual ~NetworkState(){}
};


class OpenState :public NetworkState{
public:
    static NetworkState* getInstance(){
        static NetworkState* m_instance=new OpenState();
        return m_instance;
    }

    static void destroyInstance(){
        NetworkState* instance=getInstance();
        delete instance;
    }



    void Operation1() override{
        
        //**********
        pNext = CloseState::getInstance();

        //this=new ...

      
    }
    
    void Operation2() override{
        
        //..........
        pNext = ConnectState::getInstance();
    }
    
    void Operation3() override{
        
        //$$$$$$$$$$
        pNext = OpenState::getInstance();
    }
    
    
};

class CloseState:public NetworkState{ }

class ConnectState:public NetworkState{ }
//...


//...





class NetworkProcessor{
    
    NetworkState* pState;
    
public:
    
    NetworkProcessor(NetworkState* pState){
        
        this->pState = pState;
    }

    void GotoNextState(){
        pState = pState->getNextState();
    }
    
    void OperationA() override{
        //...
        pState->Operation1();
        
        this->GotoNextState();   //pState=pState->pNext;
        //...
    }
    
    void OperationB() override{
        //...
        pState->Operation2();
       
        //...
        this->GotoNextState();
    }
    
    void OperationC() override{
        //...
        pState->Operation3();
        
        //...
        this->GotoNextState();
    }
};


