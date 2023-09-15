#include <vector>
#include <memory>
#include <map>
#include <string>
#include <iostream>
using namespace std;




struct Shape{
  
  virtual unique_ptr<Shape> clone() const=0;
  virtual ~Shape() = default;
};

template<typename T>
struct ShapePrototype:  Shape
{

   unique_ptr<Shape> clone() const override
   {
      return make_unique<T>(static_cast<const T&> (*this) ); // *this : ShapePrototype&

      // return make_unique<Rectangle>(*this);
      // return make_unique<Circle>(*this);
      // return make_unique<Line>(*this);
      
   }

};


struct Line : ShapePrototype<Line>
{
  int data;
  string text;
 
  Line()=default;
  //前提：实现正确的拷贝构造
  Line(const Line& r)=default;

  Line& operator=(const Line& r)=default;

  void draw() 
  {
    cout << "draw the line" << endl;
  }



};

struct Rectangle : ShapePrototype<Rectangle>
{
  void draw() 
  {
    cout << "draw the rectangle" << endl;
  }

 
};

struct Circle : ShapePrototype<Circle>
{
  void draw() 
  {
    cout << "draw the circle" << endl;
  }


};

int main(){

  vector< unique_ptr<Shape>> v;


  v.push_back(make_unique<Line>());
  v.push_back(make_unique<Rectangle>());
  v.push_back(make_unique<Circle>());


  
}