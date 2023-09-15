module;

#include <iostream>

module CrmDAL; //模块实现单元

using namespace std; 


void CustomerDAO::create() const
{   
    cout<<"CustomerDAO::create()"<<endl;

}

void CompanyDAO::update() const
{   
    cout<<"CompanyDAO::update()"<<endl;
    
}

void DealDAO::submit() const
{   
    cout<<"DealDAO::submit()"<<endl;
    
    
}

void OrderDAO::process() const
{   
    cout<<"OrderDAO::process()"<<endl;
    
}





