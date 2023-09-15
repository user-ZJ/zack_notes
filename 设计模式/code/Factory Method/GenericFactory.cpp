#include <memory>
#include "GenericShape.h"

using namespace std;

template<typename T>
struct ShapeFactory 
{
  template<typename... Args>
  unique_ptr<Shape> create(Args&& ... args)  {
    return make_unique<T>(std::forward<Args>(args)...);
  }
};

int main(){

    ShapeFactory<Line> s1;
    unique_ptr<Shape> p1=s1.create(10,20);
    p1->draw();

    cout<<endl;

    ShapeFactory<Rectangle> s2;
    unique_ptr<Shape> p2=s2.create(10,20,100,200);
    p2->draw();

    

}
