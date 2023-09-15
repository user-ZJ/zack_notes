#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

enum LogLevel
{
    INFO,
    WARNING,
    ERROR
};

class Logger
{
public:
    Logger(LogLevel level = INFO) : level_(level) {}

    template<typename... Args>
    void log(LogLevel level, Args... args)
    {
        if (level >= level_)
        {
            std::string message = format(args...);
            switch (level)
            {
            case INFO:
                std::cout << "[INFO] ";
                break;
            case WARNING:
                std::cout << "[WARNING] ";
                break;
            case ERROR:
                std::cout << "[ERROR] ";
                break;
            default:
                break;
            }
            std::cout << message << std::endl;
        }
    }

    void set_level(LogLevel level)
    {
        level_ = level;
    }

private:
    LogLevel level_;

    std::string format()
    {
        return "";
    }

    template<typename T, typename... Args>
    std::string format(T value, Args... args)
    {
        std::stringstream ss;
        ss << value;
        return ss.str() + " " + format(args...);
    }
};

int main()
{
    Logger logger(INFO);
    logger.log(INFO, "This is an information message.");
    logger.log(WARNING, "This is a warning message.");
    logger.log(ERROR, "This is an error message.");
    logger.log(ERROR, "The value of pi is: ", 3.1415926);
    return 0;
}