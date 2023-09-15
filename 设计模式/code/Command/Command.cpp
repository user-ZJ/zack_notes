#include <iostream>
#include <vector>
#include <string>
using namespace std;


//对象表达行为
class Command
{
public:
    virtual void execute() = 0;
    
};


class CopyCommand : public Command
{
    int id;
    string arg;
public:
    CopyCommand(const string & a) : arg(a) {}
    void execute() override
    {
        cout<< "#1 process..."<<arg<<endl;
    }
};

class CutCommand : public Command
{
    int arg;
public:
    CutCommand(const int & a) : arg(a) {}
    void execute() override
    {
        cout<< "#2 process..."<<arg<<endl;
    }
};

class PasteCommand : public Command
{
    int arg;
public:
    PasteCommand(const int & a) : arg(a) {}
    void execute() override
    {
        cout<< "#2 process..."<<arg<<endl;
    }
};
        
        
class MacroCommand : public Command
{
    vector<unique_ptr<Command>> commands;
public:
    void addCommand( unique_ptr<Command> c) 
    { commands.push_back(std::move(c)); }

    void execute() override
    {
        for (auto &c : commands)
        {
            c->execute();
        }
    }
};


void process(const Command& command)
{
    
}
        

template<typename T>
class FuctionObject{
public:
    void operator()(T data)
    {

    }
};


        
int main()
{


    auto cmd1=make_unique<CopyCommand>("Arg ###1");
    auto cmd2=make_unique<CutCommand>(100);
    
    MacroCommand macro;
    macro.addCommand(std::move(cmd1));
    macro.addCommand(std::move(cmd2));
    
    macro.execute();
    process(*cmd1);
    process(macro);


    vector<unique_ptr<Command>> vCommand;
    vCommand.push_back(make_unique<CopyCommand>("Arg ###2"));
    vCommand.push_back(make_unique<CutCommand>(200));
    

}