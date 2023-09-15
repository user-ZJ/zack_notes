C++知识点总结
========================


.. contents:: 文档目录
  :local:
  :depth: 1


工具
-------------
将C++代码进行展开： https://cppinsights.io/

Compiler Explorer 查看汇编代码：https://godbolt.org/


数据存储
------------------
* 程序数据段

  - 静态(全局)数据区:全局变量，静态变量。初始化的全局变量和静态变量在一块区域(.data)，未初始化的全局变量和未初始化的静态变量在相邻的另一块区域(.bss)
  - 堆内存：手动分配的内存。程序在运行的时候用malloc或new申请内存，程序员自己负责在何时用free 或delete释放内存。
  - 栈内存：编译器自动分配。局部变量存储在栈上，函数执行结束时这些存储单元自动被释放。
  - 常量区：编译时大小、值确定不可修改。如字符串常量等

* 程序代码段

  - 函数体。存放CPU执行的机器指令，代码区是可共享，并且是只读的

.. code:: cpp

   int a = 0;          // 全局初始化区 
   char *p1;         // 全局未初始化区
   //1） s1在静态区，"abcdef"无需额外存放，就是数组s1内部，总共占用一个串的内存
   char s1[] = "abcdef"; 
   //2）p在静态区,"abcdef",必须额外存放（在常量区，通常也在静态区），/总共占用一个指针，和一个串的内存
   const char *p ="abcdef";

   void main() 
   { 
     int b;            // 栈区
     char s[]="abcdef";//s是在栈区，“abcdef”在常量区，运行的时候复制给s，
     *s='w';//正确
     s[2]='w';//正确
     // p3在栈区，"123456"在常量区,其值不能被修改，指向常量的指针是不可以通过指针移动去修改指针所指内容的
     const char *p3 = "123456"; 
     //错误,此区域是编译的时候确定的，并且程序结束的时候自动释放的， *p3= 'w';企图修改文字常量区引起错误
     *p3='w';
     char *p2;         // 栈区
     static int c =0;         // 全局（静态）初始化区 
     p1 = (char *)malloc(10); 
     p2 = (char *)malloc(20); // 分配得来的10和20字节的区域就在堆区 
     // "123456" 放在常量区，编译器可能会将它与p3所指向的"123456"优化成一个地方 
     strcpy(p1, "123456");    
   } 


.. _inline:

inline
-----------------------
1. 用于函数内联展开。内联函数（Inline Function），又称内嵌函数或者内置函数。
2. 用于类内static成员变量初始化(C++17)

.. code-block:: cpp

    class Point{
    public:
        int x{0};
        int y{0}; 


        void print()
        {
            cout<<x<<","<<y<<endl;
        }

        inline static int data=100;
    };
    // 不使用inline则需要使用下面的方式初始化
    //int Point::data=100;

内联函数
`````````````````````
当函数比较复杂时，函数调用的时空开销可以忽略，大部分的 CPU
时间都会花费在执行函数体代码上，所以我们一般是将非常短小的函数声明为内联函数。

使用内联函数的缺点也是非常明显的，编译后的程序会存在多份相同的函数拷贝，如果被声明为内联函数的函数体非常大，
那么编译后的程序体积也将会变得很大，所以再次强调，一般只将那些短小的、频繁调用的函数声明为内联函数。

.. note:: 

    注意，要在函数定义处添加 inline 关键字，在函数声明处添加 inline关键字虽然没有错，但这种做法是无效的，编译器会忽略函数声明处的 inline关键字。

内联函数和宏定义区别
```````````````````````
1. 内联函数在运行时 **可调试** ，而宏定义不可以;
2. 编译器会对内联函数的 **参数类型做安全检查** 或自动类型转换（同普通函数），而宏定义则不会；
3. 内联函数可以访问类的成员变量，宏定义则不能；
4. 在类中声明同时定义的成员函数，自动转化为内联函数
   内联函数和普通函数相比可以加快程序运行的速度，因为不需要中断调用，在编译的时候内联函数可以直接被镶嵌到目标代码中。
   内联函数要做参数类型检查，这是内联函数跟宏相比的优势。


static
---------------

static作用是控制作用范围和数据存放在静态区。
和 **局部变量，全局变量，函数，类成员，成员函数** 组合表示如下

1. **函数体内 static 变量** 的作用范围为该函数体，不同于局部变量，
   该变量的内存只被分配一次，因此其值在下次调用时仍维持上次的值
2. 在 **模块内的 static全局变量** 可以被模块内所有函数访问，但不能被模块外其他函数访问
3. **static函数** 与普通函数作用域不同,仅在本文件。在模块内的static
   函数只可被这一模块内的其他函数调用，这个函数的使用范围被限制在声明它的模块内
4. 在 **类的static成员变量** 属于整个类所拥有，对类的所以对象只有一份拷贝。静态数据成员不能在类中初始化，一般在类外和main()函数之前初始化
   C++11之后可以使用 :ref:`inline` 关键字修饰静态成员变量在类内初始化
5. **静态成员函数** 与普通成员函数的根本区别在于：普通成员函数有 this指针，可以访问类中的任意成员；而静态成员函数没有 this
   指针，只能访问静态成员（包括静态成员变量和静态成员函数）。

const
---------------------------
在C++中，const关键字可以用来声明常量、全局变量、函数参数、类成员变量、成员函数和类对象。

1. 声明常量
    在C++中，可以使用const关键字来声明常量。常量是一个值不能被修改的表达式，定义时需要初始化
2. 全局变量
    使用const关键字修饰全局变量会使其成为常量，即该变量的值不能被修改。常量全局变量必须在定义时进行初始化，且只能被读取，不能被赋值。
3. 函数参数
    const关键字可以用于函数参数的声明，这意味着该参数在函数内部不能被修改
4. 成员变量
    const成员变量都要放在初始化列表之中进行
    
5. const成员函数
   
   * const修饰成员函数放在函数后面 `void foo() const{}`
   * const关键字可以用于成员函数的声明，以指示该函数不会修改类的任何成员变量。
   * const成员函数只能调用const成员函数，不能调用非const成员函数
   * 值得注意的是，把一个成员函数声明为const可以保证这个成员函数不修改数据成员，但是，如果据成员是指针，则const成员函数并不能保证不修改指针指向的对象，编译器不会把这种修改检测为错误。
   * const对象只能调用const成员函数,必须要提供一个const版本的成员函数
   * 如果只有const成员函数， **非const对象** 可以调用const成员函数的。当const版本和非const版本的成员函数同时出现时，非const对象调用非const成员函数。

6. 类对象
    const关键字可以用于类对象的声明。当一个类对象被声明为const时，它的成员变量不能被修改。



const和constexpr
-------------------
**constexpr** 是C++11引入的关键字，具有常量性，同时具备编译期可确定值(加强版的const);
const仅具有常量属性(不能更改)，但并不保证编译期值确定。

constexpr可以应用于一切需要编译期常量的地方：

- 数组大小
- 模板值实参
- 枚举的数值...


constexpr 不支持string，因为string有小对象优化，可能在堆上

constexpr函数
`````````````````````
声明为constexpr函数的意义是：如果其参数均为合适的编译期常量，则对这个constexpr函数的调用就可用于期望常量表达式的场合
（如模板的非类型参数，或枚举（enum）常量的值）。如果参数的值在运行期才能确定，或者虽然参数的值是编译期常量，但不匹配这个函数的要求，
则对这个函数调用的求值只能在运行期进行。

constexpr函数必须满足下述限制：

-  函数返回值不能是void类型
-  函数体不能声明变量或定义新的类型
-  函数体只能包含声明、null语句或者一段return语句
-  在形参实参结合后，return语句中的表达式为常量表达式

.. code:: cpp

   #include <iostream>
   #include <array>
   using namespace std;

   constexpr int foo(int i)
   {
       return i + 5;
   }

   int main()
   {
       int i = 10;
       std::array<int, foo(5)> arr; // OK

       foo(i); // Call is Ok

       // But...
       std::array<int, foo(i)> arr1; // Error

   }

`参考 <https://zh.wikipedia.org/wiki/Constexpr>`__


if constexpr
-----------------------
if constexpr在编译时计算条件分支

* true直接生成局部块内代码
* false则不生成代码
* 提供了比类型判断式、类型trait更好的编译时分发机制



引用
-------------
1. 引用支持多态性
2. 引用必须被初始化
3. 引用只有一级，不存在引用的引用
4. 在以下使用数据场合中：函数局部变量、参数、返回值、全局变量、类成员变量，参数情况，一般推荐使用引用。
   其他场合，都推荐使用指针


指针
--------

.. code:: cpp

    char *const pc; //常指针(不可以修改p的值)
    char const *pc1; //指向常量的指针(p所指向的内容不可修改)
    const char *pc1;  //指向常量的指针
    const int* const pc;  //指向常量的常指针
    int (*p)[4];  //数组指针，指向一个大小为4个整型的数组的数组指针
    //数组名是地址，与数组首元素地址,仅代表自己类型那么大内存不同，数组名内存指向能力非常强。
    //数组名指向整个数组空间。进一步讲，对数组名取地址，即就是在对整个数组取地址，
    //则数组的地址自然要用指向数组的指针才能接收，
    //所以，必须定义指向数组的指针类型，即为数组指针。
    int ar[10] = { 1,2,3,4,5,6,7,8,9,10 };
    int **p = &ar;  //报错
    int(*pp)[10] = &ar; //正确
    int *p[10];   //指针数组
    //函数指针，一般指针都有加1的能力，但是，函数指针不允许做这样的运算。即pfun+1是一个非法的操作
    int(*pfun)(int, int);  
    //指针函数，返回指针类型的函数称为指针函数，只要返回值为指针，无论是什么类型的指针，都称为指针函数
    int* fun(int a, int b){} 
    //返回函数指针的指针函数
    int(*func(int a, int b, int(*FUN)(int, int))) (int, int){}


引用与指针
------------------------

**指针和引用都是地址的概念，指针指向一块内存，它的内容是所指内存的地址；引用是某块内存的别名。**

引用和指针都能保持多态性

区别：

1. 引用必须被初始化，指针不必。
2. 引用初始化以后不能被改变，指针可以改变所指的对象。
3. 不存在指向空值的引用，但是存在指向空值的指针。
4. 对引用使用“sizeof”得到的是变量的大小，对指针使用“sizeof”得到的是变量的地址的大小。
5. 理论上指针的级数没有限制，但引用只有一级。即不存在引用的引用，但可以有指针的指针。
6. 就++操作而言，对引用的操作直接反应到所指向的对象，而不是改变指向；而对指针的操作，会使指针指向下一个对象，而不是改变所指对象的内容


值语义和引用语义
-------------------------
* 值语义：对象以值的方式直接存储，传参，返回值，拷贝等
* 引用语义：对象以指针或引用的方式间接存储，参数、返回值、拷贝传递的是指针或引用。

| 值语义没有悬浮指针/引用，没有昂贵的释放操作，没有内存泄漏、数据竞争。
| 值语义对大对象的拷贝代价较高，不能支持虚函数多态，不能维持对象全局唯一性




特殊成员函数与三法则
-----------------------------
* 四大特殊成员函数
  
  - 默认构造函数(无参)，如果不定义任何拷贝构造，编译器自动生成
  - 构造函数/拷贝构造函数/赋值操作符，如果不定义，编译器自动生成
  - 使用default让编译器自动生成
  - 使用delete让编译器不要自动生成

* **三法则：析构函数、拷贝构造函数、赋值操作符三者自定义其一，则需要同时定义另外两个(编译器自动生成的一般语义错误)**
* 编译器自动生成的拷贝/赋值时按字节拷贝，如果不正确，则需要自定义拷贝/赋值/析构行为。
* 需要自定义三大函数的类，通常包含指针指向的鼎泰数据成员(堆内存)

.. note:: 

    赋值操作符返回引用，是为了支持连等 “a=b=c”

支持移动语义的五法则
----------------------------------
* 如果没有自定义析构函数、拷贝构造函数、赋值操作符任何其一，编译器也会自动生成移动构造和移动赋值操作符，
  生成的是按成员进行实例成员的移动操作请求(如果不支持，则退化为拷贝)
* **如果自定义了析构函数、拷贝构造函数、赋值操作符任何其一，那么移动构造和移动赋值操作符都需要自定义，
  编译器将不再自动生成(常见陷阱的由来)**
* 如果自定义了移动构造、移动赋值操作符任何其一，编译器将不再自动生成另一个和对应的拷贝构造或赋值操作符
* 简单规则：五大特殊成员函数要么全部定义(指针指向动态数据成员)，要么全交给编译器自动生成(基本类型或对象)
* 如果类中有成员不支持拷贝(如：unique_ptr),编译器不会生成拷贝构造和拷贝赋值，但会生成移动构造和移动赋值，拷贝构造和拷贝赋值需要自己写



对齐
-----------
C++默认是四字节对齐，可以使用#pragma pack(pack_size)修改对齐字节数。pack_size表示最多以pack_size字节对齐

.. code-block:: cpp

    #pragma  pack(8)
    // code
    #pragma


类的初始化顺序
----------------------
1. 初始化列表
2. inline初始化
3. 构造函数中代码初始化


类的继承
----------------
子类继承父类，仅仅继承了实例数据成员和实例函数；不包括父类的构造器、静态数据成员、静态函数。

构造器是调用关系，子类构造器后调用父类构造器。


虚拟继承
-------------------
* 当同一个父类Base被多个子类Sub1和Sub2继承，而Sub1和Sub2又作为另外一个类Sub3的父类，就出现了所谓的"菱形继承"的结构。
  即Base在最终子类Sub3中出现了多次。
* 如果希望避免Sub3中出现多次Base，则Sub1和Sub2继承Base时，需要使用virtual继承
* virtual继承的本质是在子类中增加一个额外的指针指向基类
* 如果Base中没有数据成员(作为接口)，可以不用使用虚继承来节省内存


string
--------------
.. code-block:: cpp

    using namespace std::string_literals;
    std::string str = "hello"s;  //hello+s构建出来的是string而不是const char *


enum class
----------------------
C++11 引入了枚举类（也称为作用域枚举），这使得枚举既是强类型的又是强作用域的。枚举类不允许隐式转换为 int，也不比较来自不同枚举的枚举数。

.. code-block:: cpp

    #include <iostream>
    using namespace std;

    int main()
    {
        enum class Color { Red,Green,Blue };
        enum class Color2 { Red,Black,White };
        enum class People { Good,Bad };

        // An enum value can now be used
        // to create variables
        int Green = 10;

        // Instantiating the Enum Class
        Color x = Color::Green;

        // Comparison now is completely type-safe
        if (x == Color::Red)
            cout << "It's Red\n";
        else
            cout << "It's not Red\n";

        People p = People::Good;

        if (p == People::Bad)
            cout << "Bad people\n";
        else
            cout << "Good people\n";

        // gives an error
        // if(x == p)
        // cout<<"red is equal to good";

        // won't work as there is no
        // implicit conversion to int
        // cout<< x;

        cout << static_cast<int>(x);

        return 0;
    }

声明枚举类的枚举类型也对其底层类型有更多的控制；它可以是任何整型数据类型，例如 char、short 或 unsigned int，它们主要用于确定类型的大小。

这由枚举类型后面的冒号和基础类型指定：

.. code-block:: cpp

    #include <iostream>
    using namespace std;
    enum rainbow{
        violet,indigo,blue,green,yellow,orange,red
    }colors;
    enum class eyecolor:char{
        blue,green,brown
    }eye;
    int main() {
        cout<<"size of enum rainbow variable: "<<sizeof(colors)<<endl;
        cout<<"size of enum class eyecolor variable:"<<sizeof(eye)<<endl;
        return 0;
    }

**C++11中的枚举类型的值作为数组的长度，并使用方括号语法来初始化数组元素。**

.. code-block:: cpp

    #include <iostream>

    int main() {
        enum DataType { INT, FLOAT, DOUBLE, CHAR };
        static const int ARRAY_SIZE[DataType::CHAR + 1] = {
            [DataType::INT] = 4, 
            [DataType::FLOAT] = 4, 
            [DataType::DOUBLE] = 8, 
            [DataType::CHAR] = 1
            };
        //for (auto &v : ARRAY_SIZE)
        //    std::cout << v << " ";
        return 0;
    }



统一初始化
-------------------
大多数情况推荐使用统一初始化，又叫列表初始化，特别是对象、容器；
对于数值，可以防止隐式窄化转型。空列表{}使用默认值初始化；

* 统一初始化支持类公有成员： Point pt{100,200};
* 统一列表初始化支持容器元素： vector t{1,2,3,4,5};


结构化绑定
----------------------
C++17支持结构化绑定

* 结构化绑定支持将多个变量的值"一次性"初始化(绑定)为对象的各个 **公有** 实例成员的值(按声明顺序、即内存顺序)
* 如果使用值绑定: auto [u,v]=myObject;(会执行一次类的拷贝构造)。
  变量值是对象成员的拷贝，绑定后各自更改互不影响。
* 如果使用引用绑定：auto &[u,v]=myObject;
  变量值是对象成员的引用，绑定后各自更改会互相影响
* 结构化绑定支持类层级，数组


using用法
-------------

1. 引用基类成员
2. 导入命名空间
3. 命名空间别名，列别名
4. 模板别名

.. code:: cpp

   //using 声明 (using declaration) 是将命名空间中单个名字注入到当前作用域的机制，使得在当前作用域下访问另一个作用域下的成员时无需使用限定符 ::
   using std::map;
   // 命令空间导入
   using namespace std;
   // 命令空间别名
   using TorchModule = torch::jit::script::Module;
   using Tensor = torch::Tensor;
   namespace http = beast::http; 
   //类型重定义，取代 typedef(别名)
   using fun = void (*)(int, int);
   //typedef void (*fun)(int, int); //与上一句等价
   using int16 = short;
   //typedef short int16; //与上一句等价
   //模板别名
   template<typename T>
   using SmallVec=vector<T, SmallAlloca<T>>;//指定部分参数，模板类型

引入基类成员
`````````````````
* 函数重载不会跨越子类、父类作用域；因此如果想在子类中提供父类同名函数(即重载)，需要using Base::function;声明
* 默认情况下父类的构造函数不被子类继承，但如果希望继承，则可使用using Base::Base的方式将父类所有的构造器"继承"下来
* 由using声明引入父类的成员，其访问权限由using声明所在位置决定

.. code:: cpp

    #include <iostream>
    #include <string>
    using namespace std;
    struct Base{
        Base(){
            cout<<"Base.process()"<<endl;
        }
        Base(int data){
            cout<<"Base.process(int)"<<endl;
        }
        Base(string text){
            cout<<"Base.process(string)"<<endl;
        }
        void process()
        {
            cout<<"process()"<<endl;
        }
        void process(double data)
        {
            cout<<"process(double data)"<<endl;
        }
    };

    struct Sub:  Base{
        using Base::Base;
        using Base::process;
        //  Sub():Base(){}
        // Sub(int data):Base(data){}
        //  Sub(string text):Base(text){}
        void process(int data)
        {
            cout<<"process(int data)"<<endl;
        }
    };

    int main()
    {
        Sub s1;
        Sub s2(100);
        Sub s3("hello");
        s3.process(100);
        //s3.Base::process();
        s3.process();
        s3.process(10.23);
    }

RTTI,运行期类型识别（typeid）
----------------------------------------
* RTTI，Runtime Type Identification 运行期类型识别
* typeid运算符，返回type_info &对象引用

  * 如果对象有虚表，需要运行期获得类型信息（动态类型）
  * 如果对象没有虚表，编译时即可获得类型信息
  * 每一个类型全局对应唯一type_info对象，虚表信息指向的对象

* dynamic_cast:多态转型，父类转为子类

  * 父类必须有至少一个虚函数，才能支持多态转型
  * 转为子类指针，如果转换失败，则结果为nullptr
  * 转为子类引用，如果转换失败，则抛出异常
  * 使用运行期类型匹配（虚表类型信息查找），性能较差，慎用

.. literalinclude:: code/typeid.cpp
    :language: cpp


移动语义
------------
* 移动语义是为了解决对象深拷贝的代价。
* 移动仅复制对象本身的数据，不复制分离内存
* 拷贝既复制对象内存本身数据，也复制分离内存
* 移动永远不会比拷贝慢，通常更快
* C++通过两个操作来支持移动语义：
    
  * 移动构造函数( **T a=std::move(b);** 或者 **T a(std::move(b))**)
  * 移动赋值操作符( **T a,b; a = std::move(b)** )

* 移动构造/移动赋值参数不能是const 
* 移动构造/移动赋值需要加noexcept(对象在容器中移动的需求)
* 有 **类对象** 成员的情况下，移动构造/移动赋值函数对对象需要调用std::move进行移动
* 如果有基类，需要使用std::move调用父类移动构造/移动赋值
* :ref:`std::move`
* 不仅仅是移动构造和赋值，类中的SetXXX函数都可以使用移动重载



右值引用
```````````
* C++11引入右值引用语法：T &&
* 通常的引用 T & 被称为 **左值引用**
* 右值引用在某些方面和左值引用有类似行为(必须被初始化，不能被重新绑定)
* 对象可被移动的前提是--移动之后不在使用！这是右值的来源。(右值引用表示对象可以从这里移动到别的地方)


**左值**：

* 命名对象，可取地址，可赋值
* 字符串字面量
* 对象变量，对象成员
* 指针，指针解引用后的对象/变量
* 函数(可取地址)
* 返回左值的表达式

**右值**：

* 无名，无法获取地址，不可赋值
* 除字符串外的其他基本类型字面量
* lambda表达式
* 运算符表达式

移动注意事项
`````````````````````
1. 移动语义不会默认传递

   - 移动语义在参数上不会默认传递，右值参数是一个左值
   - void func(T &&v)函数形参v，对于函数调用者来说，v是一个右值引用参数(要传递右值给它)
   - 对于函数内部来说，v是一个左值引用，可以取地址。如果要传递给右值参数的函数，则需要调用std::move，调用之后对象失效。

2. 对const参数调用std::move不会调用移动构造
3. **原始的指针和基本类型没有移动的概念，只有类类型才有移动的概念**
4. 移动需要满足 `支持移动语义的五法则`_
5. 返回值优化 好于 移动操作 好于 拷贝(返回值优化移动操作都不需要调用)


绑定规则
`````````````````````
* 左值可以绑定到左值引用
* 左值不可以绑定到右值引用
* 右值可以绑定到左值常量引用
* 右值可以绑定到右值非常量引用

+-----------+------+--------+------+---------+
|           |  &   | const& |  &&  | const&& |
+===========+======+========+======+=========+
| 左值      | 可以 | 可以   |      |         |
+-----------+------+--------+------+---------+
| const左值 |      | 可以   |      |         |
+-----------+------+--------+------+---------+
| 右值      |      | 可以   | 可以 | 可以    |
+-----------+------+--------+------+---------+
| const右值 |      | 可以   |      | 可以    |
+-----------+------+--------+------+---------+

.. literalinclude:: code/10_binding.cpp
    :language: cpp



完美转发
`````````````
完美转发目标：同一个函数，针对左值考拷贝，针对右值移动

使用 :ref:`std::forward` 会将输入的参数原封不动地传递到下一个函数中，这个“原封不动”指的是，如果输入的参数是左值，
那么传递给下一个函数的参数的也是左值；如果输入的参数是右值，那么传递给下一个函数的参数的也是右值。

这样在参数是右值的时候可以调用对象的移动构造函数。

**完美转发只支持模板函数**

.. code:: cpp

   template<class T, class A1>
   std::shared_ptr<T> factory(A1&& a1){
       return std::shared_ptr<T>(new T(std::forward<A1>(a1));  
   }

引用折叠
`````````````````````
:: 

    & &   --> &  左值+左值 -> 左值
    & &&  --> &  左值+右值 -> 左值
    && &  --> &  右值+左值 -> 左值
    && && --> && 右值+右值 -> 右值

.. literalinclude:: code/14_ref_collapse.cpp


set_value
`````````````````
.. literalinclude:: code/17_set_value.cpp
    :language: cpp



:ref:`智能指针`
------------------------

函数对象
-------------------
* 函数对象(function object),又叫 **仿函数** 、函子(functor)。
  通过重载类的opeartor()调用操作符，实现将类对象当作函数调用的能力
* 作为类对象，函数对象可以定义实例变量，并通过构造器参数来初始化，从而使得函数对象可以携带状态数据
* 函数对象通常可以inline，其性能比函数指针要高。函数指针只能运行时辨析地址，进行间接调用，也无法内联，性能较差。
* 函数对象可以采用类模板的方式模板化，从而使得函数对象可以参与泛型编程。

auto
---------------
auto有以下几种用法：

1. 类型自动推导
    auto可以用于自动推导变量的类型
2. 迭代器自动推导
    auto可以用于自动推导迭代器的类型
3. 函数返回值类型自动推导

.. code-block:: cpp

    auto add(int x, int y) {
        return x + y;
    }

4. auto关键字结合lambda表达式

.. code-block:: cpp

    auto func = [](int x, int y) -> int { return x + y; };

5. auto关键字结合范围for循环
6. auto关键字结合decltype类型推导

.. code-block:: cpp

    int arr[] = {1, 2, 3};
    decltype(auto) c = arr; // c的类型被推导为int (&)[3]


auto和decltype
----------------------------
* 当变量有合适的初始化器时，可以直接使用auto
* 但有时候即希望编译器自动推断类型，又不希望、或者无法定义初始化变量，就应该使用decltype

  * 返回值类型依赖于形参类型的函数模板
  * decltype(auto)可以从初始化表达式中推导出来类型

* decltype(expr)推断的结果是expr的声明类型。注意 当类型为T的左值表达式，decltype的推断类型为T&

:ref:`lambda表达式`
--------------------------

函数适配器
-----------------
* 函数适配器：接受函数参数，返回可调用该函数的函数对象。本质是函数对象
* bind()使用额外实参来绑定任意函数
* mem_fun()绑定成员函数，适配为非成员函数(额外实参，仍需要bind)



编译时计算支持(C++20)
-----------------------
* constinit，变量--编译时初始化
* consteval,函数--编译时评估

constinit
`````````````````
* constinit = constexpr - const，必须在编译时初始化，但是不要求常量，可以在之后运行时改变
* constinit支持：全局变量、静态变量；字面量类型、使用字面编译时构造的对象，thread_local等提高初始化性能。
* constinit不支持：普通函数的局部变量、无编译时构造器的类对象
* 不能使用一个constinit变量去初始化另一个constinit变量
* 声明静态成员时，并不隐含inline
* 使用constinit可以解决全局成员或者静态成员的初始化失序问题

consteval
```````````````````
* constexpr即支持编译时函数，也支持运行时函数；依赖上下文需求
* consteval强制要求函数进行编译时评估
* consteval只能在编译时上下文调用、并在编译时给出结果
* consteval不能在运行时调用
* consteval也支持lambda表达式
* consteval结果可以应用于运行时

consteval函数约束

* 函数和返回值类型必须支持字面量类型
* 函数体内只能包含字面量变量
* 不能使用goto和label
* 不能是构造器或析构器，不可以有虚基类
* 不能是协程
* 可以使用堆内存，前提是编译时必须释放

final
---------
在C++11中，final是一个关键字，用于防止类被继承和虚函数被重写。

1. final修饰类，表示类不能被继承
2. final修饰虚函数，表示方法不能被overide



explicit
------------

C++中，一个参数的构造函数(或者除了第一个参数外其余参数都有默认值的多参构造函数)，承担了两个角色。

1. 是个构造；
2. 是默认且隐含的类型转换操作符。

.. code:: cpp

   #include <iostream>
   using namespace std;
   class Test1
   {
     public :
       Test1(int num):n(num){}
     private:
       int n;
   };
   class Test2
   {
     public :
       explicit Test2(int num):n(num){}
     private:
       int n;
   }; 
   int main()
   {
       Test1 t1 = 12;  //调用构造函数
       Test2 t2(13);
       Test2 t3 = 14;    //报错
       return 0;
   }

**explicit的作用是用来声明类构造函数是显示调用的**，而非隐式调用，所以只用于修饰单参构造函数。
因为无参构造函数和多参构造函数本身就是显示调用的。

当类的声明和定义分别在两个文件中时，explicit只能写在在声明中，不能写在定义中。

volatile
------------

volatile关键字是一种类型修饰符，用它声明的类型变量表示不可以被某些编译器未知的因素更改；
遇到这个关键字声明的变量， **编译器对访问该变量的代码就不再进行优化** ，从而可以提供对特殊地址的稳定访问。

.. code:: cpp

   #include <stdio.h>

   void main()
   {
       volatile int i = 10;
       int a = i;

       printf("i = %d", a);
       __asm {
           mov dword ptr [ebp-4], 20h
       }

       int b = i;
       printf("i = %d", b);
   }

一般说来，volatile用在如下的几个地方：

1. 中断服务程序中修改的供其它程序检测的变量需要加 volatile；
2. 多任务环境下各任务间共享的标志应该加 volatile；
3. 存储器映射的硬件寄存器通常也要加 volatile说明，因为每次对它的读写都可能由不同意义；

在 **多线程** 中，有些变量是用 volatile关键字声明的。当两个线程都要用到某一个变量且该变量的值会被改变时，应该用
volatile声明，该关键字的作用是 **防止优化编译器把变量从内存装入 CPU寄存器中** 。
如果变量被装入寄存器，那么两个线程有可能一个使用内存中的变量，一个使用寄存器中的变量，这会造成程序的错误执行。
volatile的意思是让编译器每次操作该变量时一定要从内存中真正取出，而不是使用已经存在寄存器中的值


assert
--------------------------
在C++中，assert 宏用于在运行时进行断言检查，如果断言条件为假（即 false），则会触发断言失败，导致程序终止。
通常，在使用 assert 时，你可能希望输出一些信息，以便在断言失败时更好地理解问题的原因。

默认情况下，assert 宏只会输出一个简单的错误信息，通常包含文件名、行号和断言条件。
然而，你可以通过在断言失败时使用输出流来输出自定义的信息。一个常用的方法是使用 std::cerr 流来输出错误信息。

.. code-block:: cpp

    #include <iostream>
    #include <cassert>

    int main() {
        int x = 5;
        int y = 10;
        assert(x == y && "x and y are not equal!");
        std::cout << "Continuing after assert..." << std::endl;
        return 0;
    }




:ref:`类型转换`
-----------------


:ref:`模板编程`
------------------------


:ref:`协程`
--------------


:ref:`其他常问问题`
-----------------------------
