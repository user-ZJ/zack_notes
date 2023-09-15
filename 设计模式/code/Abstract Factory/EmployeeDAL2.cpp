
//数据库访问有关的基类
class IDBConnection{
    
};

class IDBConnectionFactory{
public:
    virtual unique_ptr<IDBConnection> CreateDBConnection()=0;
};


class IDBCommand{
    
};
class IDBCommandFactory{
public:
    virtual unique_ptr<IDBCommand> CreateDBCommand()=0;
};


class IDataReader{
    
};
class IDataReaderFactory{
public:
    virtual unique_ptr<IDataReader> CreateDataReader()=0;
};


//支持MySQL
class SqlConnection: public IDBConnection{
    
};
class SqlConnectionFactory:public IDBConnectionFactory{
public:
    unique_ptr<IDBConnection> CreateDBConnection() override
    {

    }
    
};


class SqlCommand: public IDBCommand{
    
};
class SqlCommandFactory:public IDBCommandFactory{
public:
    unique_ptr<IDBCommand> CreateDBCommand() override
    {

    }
};


class SqlDataReader: public IDataReader{
    
};
class SqlDataReaderFactory:public IDataReaderFactory{
public:
    unique_ptr<IDataReader> CreateDataReader() override
    {

    }   
};




//支持Oracle
class OracleConnection: public IDBConnection{
    
};

class OracleConnectionFactory:public IDBConnectionFactory{
public:
    unique_ptr<IDBConnection> CreateDBConnection() override
    {

    }
    
};


class OracleCommand: public IDBCommand{
    
};

class OracleCommandFactory:public IDBCommandFactory{
public:
    unique_ptr<IDBCommand> CreateDBCommand() override
    {

    }
};


class OracleDataReader: public IDataReader{
    
};
class OracleDataReaderFactory:public IDataReaderFactory{
public:
    unique_ptr<IDataReader> CreateDataReader() override
    {

    }   
};



class EmployeeDAO{

    unique_ptr<IDBConnectionFactory> dbConnectionFactory;// SqlConnectionFactory
    unique_ptr<IDBCommandFactory> dbCommandFactory;//OracleCommandFactory
    unique_ptr<IDataReaderFactory> dataReaderFactory;

    EmployeeDAO()
    {
        dbConnectionFactory=....;//SqlConnectionFactory
        dbCommandFactory=....;//OracleCommandFactory
        dataReaderFactory=....;
    }
    
public:
    vector<EmployeeDO> GetEmployees(){
        unique_ptr<IDBConnection> connection =
            dbConnectionFactory->CreateDBConnection();
        connection->ConnectionString("...");

        unique_ptr<IDBCommand> command =
            dbCommandFactory->CreateDBCommand();
        command->CommandText("...");
        command->SetConnection(std::move(connection)); //关联性

        unique_ptr<IDBDataReader> reader = 
            command->ExecuteReader(); //关联性
        while (reader->Read()){

        }

    }

    void UpdateEmployee()
    {

    }

    void DeleteEmployee()
    {

    }
};
