#include <iostream>

using  namespace std;

import CrmDAL; // 导入模块



int main()
{
    CustomerDAO custDAO;
    custDAO.create();

    CompanyDAO comDAO;
    comDAO.update();

    DealDAO dealDAO;
    dealDAO.submit();

    OrderDAO orderDAO;
    orderDAO.process();




    
}