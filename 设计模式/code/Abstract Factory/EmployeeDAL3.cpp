
//数据库访问有关的基类
class IDBConnection{   
};
class IDBCommand{   
};
class IDataReader{  
};



class IDBFactory{   //系列工厂 MySQL -> Oracle-> SQL Server...
public:
    virtual unique_ptr<IDBConnetion> CreateDBConnection()=0;
    virtual unique_ptr<IDBCommand> CreacteDBCommand()=0;
    virtual unique_ptr<IDataReader> CreateDataReader()=0;

};


//支持My SQL
class SqlConnection: public IDBConnection{
    
};
class SqlCommand: public IDBCommand{
    
};
class SqlDataReader: public IDataReader{
    
};


class SqlDBFactory:public IDBFactory{
public:
    unique_ptr<IDBConnection> CreateDBConnection() override
    {

    }

    unique_ptr<IDBCommand> CreateDBCommand() override
    {

    }

    unique_ptr<IDataReader> CreateDataReader() override
    {

    }   
 
};

//支持Oracle
class OracleConnection: public IDBConnection{
    
};

class OracleCommand: public IDBCommand{
    
};

class OracleDataReader: public IDataReader{
    
};

class OracleDBFactory:public IDBFactory{
public:
    unique_ptr<IDBConnection> CreateDBConnection() override
    {

    }

    unique_ptr<IDBCommand> CreateDBCommand() override
    {

    }

    unique_ptr<IDataReader> CreateDataReader() override
    {

    }   
 
};



//.........................

class EmployeeDAO{
    unique_ptr<IDBFactory> dbFactory;// SqlDBFactory, OracleDBFactory...
    
public:

    EmployeeDAO(unique_ptr<IDBFactory> dbFactory):dbFactory(std:move(dbFactory))
    {
        
    }


    vector<EmployeeDO> GetEmployees(){
        unique_ptr<IDBConnection> connection =
            dbFactory->CreateDBConnection();
        connection->ConnectionString("localhost...");

        unique_ptr<IDBCommand> command =
            dbFactory->CreateDBCommand();
        command->CommandText("Select...");
        command->SetConnection(std::move(connection)); //关联性

        unique_ptr<IDBDataReader> reader =
            dbFactory->CreateDataReader();
        reader->SetCommand(std::move(command));//关联性
        while (reader->Read()){

        }

    }


};
