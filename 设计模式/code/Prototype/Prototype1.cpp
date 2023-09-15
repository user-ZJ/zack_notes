#pragma once
#include <memory>
#include <map>
#include <string>
using namespace std;






// Prototype
struct Shape
{
 
  virtual ~Shape() = default;

  virtual void draw() = 0;

  virtual unique_ptr<Shape> clone()=0;

  

};



struct Line : Shape
{
  int data;
  string text;
 
  //前提：实现正确的拷贝构造
  Line(const Line& r)=default;
  Line& operator=(const Line& r)=default;

  void draw() override
  {
    cout << "draw the line" << endl;
  }

   unique_ptr<Shape> clone() const override
   {

      // unique_ptr<Shape> p {new Line(*this))} ; //深拷贝、浅拷贝？
      // return p;

      return make_unique<Line>(*this);
   }

};

struct Rectangle : Shape
{
  void draw() override
  {
    cout << "draw the rectangle" << endl;
  }

  
   unique_ptr<Shape> clone() const override
   {
      // unique_ptr<Shape> p {new Rectangle(*this))} ; //深拷贝、浅拷贝？
      // return p;

      return make_unique<Rectangle>(*this);
   }
};

struct Circle : Shape
{
  void draw() override
  {
    cout << "draw the circle" << endl;
  }

   unique_ptr<Shape> clone() const override
   {
      // unique_ptr<Shape> p {new Circle(*this))} ; //深拷贝、浅拷贝？
      // return p;

      return make_unique<Circle>(*this);
   }
};