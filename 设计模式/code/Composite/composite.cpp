#include <iostream>
#include <list>
#include <string>
#include <algorithm>
#include <memory>

using namespace std;



class Control {


 public:
  virtual ~Control() {}

  virtual bool isComposite() {
    return false;
  }

  
  virtual void process()  = 0;



};



class Label : public Control {


 public:
  void process() override {
        cout<< "...Label" <<endl;
  }

   virtual ~Label() {
        cout<<"~Lable"<<endl;
   }


};

class Textbox : public Control {

 public:
    void process() override {
        cout<< "...Textbox" <<endl;
    }
    virtual ~Textbox() {
        cout<<"~Textbox"<<endl;
    }
};

class Button : public Control {
 public:
    void process() override {
        cout<< "...Button" <<endl;
    }
      virtual ~Button() {
        cout<<"~Button"<<endl;
    }
};


class ControlComposite : public Control {

 protected:
   list<shared_ptr<Control>> children_;

   //list<Control> children_;
   //list<Control*> children_;
   //list<unique_ptr<Control>> children_;
   //list<weak_ptr<Control>> children_;

    list<shared_ptr<Control>> getChildren()
    {
        return children_;
    }

 public:
 
 

    void add( const shared_ptr<Control>& component)  {
        this->children_.push_back(component);

    
    }
    
    void remove(const shared_ptr<Control>& component)  {
        children_.remove(component);
    }

    bool isComposite()  override {
        return true;
    }
    
    void process() override {
        
        //1.处理当前树节点
        cout<<"...ControlComposite"<<endl;

        //2.处理所有子节点
        for ( auto& c : children_) {
            c->process(); //多态调用
        }
    }

    virtual ~ControlComposite() {
        cout<<"~ControlComposite"<<endl;
    }
};

// 不用Composite模式时的处理
/*void invoke(const Control& item)
{
    if(item.isComposite())
    {
        const ControlComposite& composite=dynamic_cast<const ControlComposite&>(item);
        list<shared_ptr<Control>> list=composite.getChildren();
        for(auto& c: list)
        {
            invoke(*c);
        }
    }
    else {
        item.process();
    }
}*/



void invoke( Control& item)
{

    //...
    item.process();

    //....
}

int main() {

   
    auto composite=make_shared<ControlComposite>();

    {
        auto c1= make_shared<Label>();
        auto c2= make_shared<Textbox>();
        auto c3= make_shared<Button>();

        composite->add(c1);
        composite->add(c2);
        composite->add(c3);

        auto composite2=make_shared<ControlComposite>();
        composite2->add(c2);
        composite->add(composite2);

    }

    cout<<"------process start"<<endl;

    composite->process();

    invoke(*composite);

    cout<<"------process end"<<endl;
}
