boost使用笔记
=============================

字符串格式化
---------------------
.. code-block:: cpp

    #include <iostream>
    #include <boost/format.hpp>

    int main() {
        std::string name = "Alice";
        int age = 20;
        double height = 1.75;
        // 宽字符用boost::wformat
        std::string s = boost::str(boost::format("My name is %1%, I'm %2% years old, 
                        and my height is %3% meters.") % name % age % height);
        std::cout << s << std::endl;
        return 0;
    }