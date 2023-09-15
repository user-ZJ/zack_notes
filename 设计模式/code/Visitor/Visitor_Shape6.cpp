#include <iostream>
#include <memory>
#include <vector>
#include <variant>
#include <algorithm>

using namespace std;



class Circle  {
    string name_;
public:
    Circle(const string& name) : name_(name) {}
    const string& name() const { return name_; }

};

class Triangle {
    string name_;
    double x,y,z;
public:
    Triangle(const string& name) : name_(name) {}
    const string& name() const { return name_; }

};

using CirclePtr=unique_ptr<Circle>;
using TrianglePtr=unique_ptr<Triangle>;

using ShapePtr = std::variant<CirclePtr, TrianglePtr>;

template<typename VisitorPolicy>
void process(vector<ShapePtr>& v, VisitorPolicy  visitor)
{
    for_each( v.begin(), v.end(), 
      [=](auto& elem) mutable { 
        std::visit(visitor, elem);
      }
    );
}

template<class... Ts> 
struct overloaded : Ts... { 
    using Ts::operator()...;  //继承父类所有operator() 操作符
};

template<class... Ts> 
overloaded(Ts...) -> overloaded<Ts...>; //自定义模板推导




int main(){

    vector<ShapePtr> v;
    v.push_back(make_unique<Circle>("pie"));
    v.push_back(make_unique<Triangle>("hill"));
    v.push_back(make_unique<Circle>("ring"));
 
    cout<<sizeof(ShapePtr)<<endl;
  
    auto drawingVisitor=overloaded{
        [](const CirclePtr& c) { cout << "Draw : " << c->name() << " circle" << endl;  }, 
        [](const TrianglePtr& d) { cout << "Draw : " << d->name() << " triangle" << endl; }
    };
    process(v, drawingVisitor);
    cout<<endl;

    auto zoomingVisitor=overloaded{
        [](const CirclePtr& c) { cout << "Zoom : " << c->name() << " circle" << endl;  }, 
        [](const TrianglePtr& d) { cout << "Zoom : " << d->name() << " triangle" << endl; }
    };
    process(v,zoomingVisitor);
}

