#include <iostream>
#include <string>

using namespace std;

//int globle=100;

class Singleton{
private:

    Singleton(int data):m_data(data){

        //访问共享数据
        //...
        //globle++;
    }

     Singleton(const Singleton& rhs)=delete;
     Singleton& operator=(const Singleton& rhs)=delete;
public:
    void print(){
        cout<<m_data<<endl;
    }

    static Singleton& getSingleton();
private:
    int m_data;
};



Singleton& Singleton::getSingleton()
{
    static Singleton instace{100}; //C++11之后 
    return instace;
}


int main(){

    Singleton& s1=Singleton::getSingleton();
    Singleton& s2=Singleton::getSingleton();

    cout<<&s1<<endl;
    cout<<&s2<<endl;

}