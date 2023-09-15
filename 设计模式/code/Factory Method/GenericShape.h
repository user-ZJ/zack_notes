
#include <string>
#include <iostream>

using namespace std;

struct Shape
{
  virtual ~Shape() = default;

  virtual void draw() = 0;

 
};

struct Line : Shape
{
  Line(int x, int y)
  {
    cout<< "Line(int x, int y)"<<endl;
  }

  void draw() override
  {
    cout << "draw the line" << endl;
  }

};

struct Rectangle : Shape
{
  Rectangle(int x, int y, int h, int w)
  {
    cout<< "Rectangle(int x, int y, int h, int w)"<<endl;
  }
  void draw() override
  {
    cout << "draw the rectangle" << endl;
  }
};

struct Circle : Shape
{
  void draw() override
  {
    cout << "draw the circle" << endl;
  }
};