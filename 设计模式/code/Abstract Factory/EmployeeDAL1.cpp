
class SqlConnection{};

class SqlCommand{};

class SqlDataReader{};

class EmployeeDAO{
public:
    vector<EmployeeDO> GetEmployees(){
        
        unique_ptr<SqlConnection> connection=
            make_unique<SqlConnection>();

        connection->ConnectionString ("...");
        connection->open();

        unique_ptr<SqlCommand> command=
            make_unique<SqlCommand>();
        
        command->CommandText("select...");
        command->SetConnection(std::move(connection)); //关联性

        vector<EmployeeDO> employees;

        unique_ptr<SqlDataReader> reader = 
            make_unique<SqlDataReader>();
        reader->SetCommand(std::move(command));//关联性
        while (reader->Read()){
            //...
        }

    }

    void UpdateEmployee()
    {

    }

    void DeleteEmployee()
    {

    }
};
