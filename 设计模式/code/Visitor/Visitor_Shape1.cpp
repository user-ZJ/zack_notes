#include <iostream>
#include <vector>
#include <memory>

using namespace std;

class Circle;
class Triangle;

class ShapeVisitor {
public:
    virtual void visit(Circle* c) =0;
    virtual void visit(Triangle* d) =0;

};


class Shape {
public:
    virtual ~Shape() {}
    Shape(const std::string& color) : color_(color) {}
    const std::string& color() const { return color_; }

    virtual void accept(ShapeVisitor& v) = 0;

    void accept(ShapeVisitor& v)  { 
        v.visit(this); 
    }

private:
    std::string color_;
};


class Circle : public Shape {
public:
    Circle(const std::string& color) : Shape(color) {}
    
    void accept(ShapeVisitor& v) override { 
        v.visit(this); 
    }
};

class Triangle : public Shape {
    public:
    Triangle(const std::string& color) : Shape(color) {}
    
    void accept(ShapeVisitor& v) override { 
        v.visit(this); 
    }
};

//=====================




class DrawingVisitor : public ShapeVisitor {
public:
    void visit(Circle* c) override { 
        std::cout << "Draw to the " << c->color() << " circle" << std::endl; }
   
    void visit(Triangle* d) override { 
        std::cout << "Draw to the " << d->color() << " triangle" << std::endl; }
};


class ZoomingVisitor : public ShapeVisitor {
public:
    void visit(Circle* c) override { std::cout << "Zoom the " << c->color() << " circle" << std::endl; }
    void visit(Triangle* d) override {  }
};

int main() {

    vector<unique_ptr<Shape>> v;
    v.push_back( make_unique<Circle>("pie"));
    v.push_back( make_unique<Triangle>("hill"));
    
    Circle c("pie");
    Triangle d("hill");

    DrawingVisitor fv;
    c.accept(fv);// 双重分发 dynamic dispatch
    d.accept(fv);

    ZoomingVisitor pv;
    c.accept(pv);
    d.accept(pv);
}
