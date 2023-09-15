module;

#include <iostream>

export module CrmDAL; // 声明模块Complex

using namespace std; 

//模块对外接口
export {

    class CustomerDAO{
    public:

        void create() const;
    };

    class CompanyDAO{
    public:

        void update() const;
    };

    class DealDAO{
    public:

        void submit() const;
    };

    class OrderDAO{
    public:

        void process() const;
    };

}

class DBConnection{

};

class DBCommand{

};
class DBDataReader{

};
