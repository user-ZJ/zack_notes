#include <iostream>
#include <memory>

using namespace std;


template<typename Derived>
class Shape {
    string name_;
    public:
    Shape(const string& name) : name_(name) {}
    const string& name() const { return name_; }

    template < typename Visitor>
    void accept(Visitor& v) { 
        v.visit(static_cast<Derived&>(*this));
    }
};


class Circle : public Shape<Circle> {
    public:
    using Shape::Shape;
};

class Triangle : public Shape<Triangle> {
    public:
    using Shape::Shape;
};

//===========================

class DrawingVisitor {
    public:
    void visit(Circle& c) { 
        cout << "Draw : " << c.name() << " circle" << endl; 
    }
    void visit(Triangle& d) { 
        cout << "Draw : " << d.name() << " triangle" << endl; 
    }
};

class ZoomingVisitor {
    public:
    void visit(Circle& c) { 
        cout << "Zoom : " << c.name() << " circle" << endl; 
    }
    void visit(Triangle& d) { 
        cout << "Zoom : " << d.name() << " triangle" << endl; 
    }
};

int main() {

    vector<unique_ptr<Shape<?>>> v;

    v.push_back( make_unique<Circle>("pie"));
    v.push_back( make_unique<Triangle>("hill"));
    
    Circle c("pie");
    Triangle t("hill");

    DrawingVisitor dv;
    c.accept(dv);
    t.accept(dv);

    ZoomingVisitor zv;
    c.accept(zv);
    t.accept(zv);
}
