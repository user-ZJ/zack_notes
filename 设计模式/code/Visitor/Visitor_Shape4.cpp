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



using Shape = std::variant<Circle, Triangle>;



template<class... Ts> 
struct overloaded : Ts... { 
    using Ts::operator()...;  //继承父类所有operator() 操作符
};

template<class... Ts> 
overloaded(Ts...) -> overloaded<Ts...>; //自定义模板推导




int main(){

    vector<Shape> v;
    v.push_back(Circle{"pie"});
    v.push_back(Triangle{"hill"});
    v.push_back(Circle{"ring"});

    
    cout<<sizeof(Circle)<<";"<<sizeof(Triangle)<<";"<<sizeof(Shape)<<endl;
  
    auto drawingVisitor=overloaded{
        [](const Circle& c) { cout << "Draw : " << c.name() << " circle" << endl;  }, 
        [](const Triangle& d) { cout << "Draw : " << d.name() << " triangle" << endl; }
    };

    for_each( v.begin(), v.end(), 
      [=](auto& elem) mutable { 
        std::visit(drawingVisitor, elem);
      }
    );

    cout<<endl;

    auto zoomingVisitor=overloaded{
        [](const Circle& c) { cout << "Zoom : " << c.name() << " circle" << endl;  }, 
        [](const Triangle& d) { cout << "Zoom : " << d.name() << " triangle" << endl; }
    };

    for_each( v.begin(), v.end(), 
      [=](auto& elem) mutable { 
        std::visit(zoomingVisitor, elem);
      }
    );
}

