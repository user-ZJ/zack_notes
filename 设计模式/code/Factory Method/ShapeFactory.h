#include "Shape.h"



struct ShapeFactory
{

    //virtual Shape create()=0;
    //virtual Shape* create()=0;
    //virtual Shape& create()=0;
    virtual unique_ptr<Shape> create()=0;
    virtual ~ShapeFactory() {}
};


struct LineFactory : ShapeFactory
{
  unique_ptr<Shape> create()  override {
    return make_unique<Line>();

    // Line* p= new Line();
    // auto pLine=unique_ptr<Line>(p);
    // return pLine;
  }
};

struct RectangleFactory : ShapeFactory
{
  unique_ptr<Shape> create()  override {
    return make_unique<Rectangle>();
  }
};

struct CircleFactory : ShapeFactory
{
  unique_ptr<Shape> create()  override {
    return make_unique<Circle>();
  }
};




//工厂使用
class Client{

    unique_ptr<ShapeFactory> pFactory;

  public:
    Client(){

      pFactory=....
    }

    void process()
    {
      auto pShape=pFactory->create();


    }

    void foo()
    {
      auto pShape=pFactory->create();


    }

};