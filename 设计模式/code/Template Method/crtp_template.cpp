#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
using namespace std;



template <typename T> 
class Library
{
public:
	void run() //template method
	{
		step1();
		while (!sub()->step2()) //子类调用
			step3();
		sub()->step4(); //子类调用
		step5();
	}
  
  void destroy() 
  { delete sub(); }


protected: 

	T* sub(){
		return static_cast<T*>(this); 
	}
	void step1() 
	{
		cout << "Library.step1()"<<endl;
	}

	 void step3() 
	{
		cout << "Library.step3()"<<endl;
	}

	 void step5()
	{
		cout << "Library.step5()"<<endl;
	}
	int number{0};

	
public:
	 bool step2() { return false;}
	 int step4() { return 0;}

   ~Library(){
    cout<<"Library dtor"<<endl;
  }
};

//============================

class App : public Library<App>
{
public:
	bool step2() 
	{
      cout<<"App.step2()"<<endl;
      number++;
	  return number>=4;
	}
	int step4() 
	{
      cout<<"App.step4() : "<<number<<endl;
	  return number;
	}

  ~App(){
    cout<<"App dtor"<<endl;
  }

};






int main()
{

	{
		Library<App> *pLib=new App();

		pLib->run();

		pLib->destroy();

		cout<<endl;

	}
	
	cout<<endl;
 
	{
		auto lambda = []( auto p) { p->destroy(); };
		unique_ptr<Library<App>, decltype(lambda)> uptr(new App(), lambda);
		uptr->run();
	}



}

