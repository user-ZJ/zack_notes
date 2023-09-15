#include <iostream>
#include <memory>
#include <vector>

using namespace std;



class Circle  {
    string name_;
public:
    Circle(const string& name) : name_(name) {}
    const string& name() const { return name_; }

};

class Triangle {
    string name_;
public:
    Triangle(const string& name) : name_(name) {}
    const string& name() const { return name_; }

};

struct DrawingVisitor {

    void operator()(const Circle& c) { 
        cout << "Draw : " << c.name() << " circle" << endl; 
    }

    void operator()(const Triangle& d) { 
        cout << "Draw : " << d.name() << " triangle" << endl; 
    }
};

struct ZoomingVisitor {

    void operator()(const Circle& c) { 
        cout << "Zoom : " << c.name() << " circle" << endl; 
    }
    void operator()(const Triangle& d) { 
        cout << "Zoom : " << d.name() << " triangle" << endl; 
    }
};



using Shape = std::variant<Circle, Triangle>;

template<typename VisitorPolicy>
void process(vector<Shape>& v, VisitorPolicy  visitor)
{
    for_each( v.begin(), v.end(), 
      [=](Shape& elem) mutable { 

        std::visit(visitor, elem);
        
      }
    );
}


int main(){

    vector<Shape> v;
    v.push_back(Circle{"pie"});
    v.push_back(Triangle{"hill"});
    v.push_back(Circle{"ring"});
    
    process(v, DrawingVisitor{});




}

