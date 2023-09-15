#include <memory>
#include <map>
#include <string>
#include <iostream>

using namespace std;

class Widget
{
public :
    Widget( string const& id ) : id( id )
    {
        cout<<"ctor : #"<<id<<endl;
    }

    ~Widget()
    {
        cout<<"dtor : #"<<id<<endl;
    }



    string id ;
    // other data...


};


class WidgetFactory
{
public :
    shared_ptr< Widget > get( string const& id )
    {
        map<string, weak_ptr<Widget>>::iterator x = mData.find( id );

        if( x != mData.end() )
        {
            shared_ptr<Widget> widget = x->second.lock();

            if (widget!=nullptr)
            {
                return widget ;
            }
        }


        shared_ptr<Widget> shareWidget( new Widget( id ) );
        weak_ptr<Widget > weakWidget( shareWidget );
        mData[id]= weakWidget;


        return shareWidget ;
    }

 

    map<string, weak_ptr<Widget>> mData ;
};


int main()
{


    WidgetFactory widgetFactory ;

    
    shared_ptr< Widget > w1= widgetFactory.get( "BJ001" );
    shared_ptr< Widget > w2 = widgetFactory.get( "SH002" );
    shared_ptr< Widget > w3 = widgetFactory.get( "SZ003" );


    w1.reset();
    w2.reset();
    w3.reset();

 

    shared_ptr< Widget > w4 = widgetFactory.get( "SH002" );

    w4.reset();
 
    


    

}