#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

using namespace std;

class Circle;
class Triangle;

class ShapeVisitor {
    public:
    virtual void visit(Circle& c) = 0;
    virtual void visit(Triangle& d) = 0;
};


class Shape {
public:
    virtual ~Shape() {}
    Shape(const string& name) : name_(name) {}
    const string& name() const { return name_; }
    virtual void accept(ShapeVisitor& v) = 0;
private:
    string name_;
};

template <typename Derived>
class Visitable : public Shape {
    public:
    using Shape::Shape;
    void accept(ShapeVisitor& v) override {
        v.visit(static_cast<Derived&>(*this)); 
    }
};

class Circle : public Visitable<Circle> {
    using Visitable<Circle>::Visitable;




};

class Triangle : public Visitable<Triangle> {
    using Visitable<Triangle>::Visitable;

};

class DrawingVisitor : public ShapeVisitor {
    public:
    void visit(Circle& c) override { 
        cout << "Draw : " << c.name() << " circle" << endl; }
    void visit(Triangle& d) override { 
        cout << "Draw : " << d.name() << " triangle" << endl; 
    }
};

class ZoomingVisitor : public ShapeVisitor {
    public:
    void visit(Circle& c) override { 
        cout << "Zoom : " << c.name() << " circle" << endl; 
    }
    void visit(Triangle& d) override { 
        cout << "Zoom : " << d.name() << " triangle" << endl; 
    }
};

int main() {

    vector<unique_ptr<Shape>> v;
    v.push_back( make_unique<Circle>("pie"));
    v.push_back( make_unique<Triangle>("hill"));
    v.push_back( make_unique<Circle>("ring"));
 
    DrawingVisitor dv;

    ZoomingVisitor zv;

    for_each( v.begin(), v.end(), 
      [=](auto& elem) mutable { 
        elem->accept(dv); // double dynamic dispatch
      }
    );
}
