===========
STL使用笔记
===========

vector
---------

*vector*\ 是一个能够存放任意类型的动态数组，能够增加和压缩数据，增加的时候容量不够，则以2倍的存储增加容量

https://blog.csdn.net/phoebin/article/details/3864590

创建
^^^^^^^^

.. code-block:: cpp

    #include <vector>
    int main(int argc,char *argv[]){
        std::vector<int> vInts;  //创建空vector
        std::vector<int> vInts(100);  //创建容量为100的vector
        std::vector<int> vInts(100,1); //创建容量为100的vector，并全部初始化为1
        std::vector<int> vInts(vInts1);  //拷贝一个vector内容来创建一个vector
        std::vector<int> vInts{1,2,3,4,5}; //  
        vInts.reserve(100);  //新元素还没有构造,此时不能用[]访问元素
        //子序列
        std::vector<int> sub_vec(int_vec.begin(), int_vec.begin()+5);
        std::vector<int> sub_vec = {int_vec.begin(), int_vec.begin()+5};
        //创建矩阵
        std::vector<std::vector<float>> result(10,vector<float>(20,0.0f));
    }

增加/插入数据
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include <vector>
   using std::vector;
   vector<int> vInts;
   vInts.push_back(9);  //在尾部增加数据
   vInts.insert(vInts.begin()+1,6); //插入数据
   a.insert(a.end(), b.begin(), b.end()); //将vector b append到vector a后面

获取容器大小
^^^^^^^^^^^^^^^^

.. code-block:: cpp

    #include <vector>
    int main(int argc,char *argv[]){
        std::vector<int> vInts;
        vInts.empty();   //判断是否为空
        vInts.size();   //返回容器中实际数据的个数。
        vInts.capacity();  // 获取vector的容量
        vInts.reserve(10); // 将容量设置为10
        
    }
   

访问数据
^^^^^^^^^^^^

.. code-block:: cpp

   #include <vector>
   using std::vector;
   vector<int> vInts(10,9);
   vInts.at(2);   //推荐使用，at()进行了边界检查，如果访问超过了vector的范围，将抛出一个异常
   vInts[2];    //不推荐使用，主要是为了与C语言进行兼容。它可以像C语言数组一样操作
   vInts.back(); // 获取末尾数据
   vInts.front(); // 获取第一个数据

删除数据
^^^^^^^^^^^^

.. code-block:: cpp

   #include <vector>
   using std::vector;
   vector<int> vInts(10,9);
   vInts.erase(vInts.begin()+3);  //删除pos位置的数据
   vInts.erase(vInts.begin(),vInts.end());  //删除pos位置的数据
   vInts.pop_back();  //删除最后一个数据。
   vInts.clear()();  //删除所有数据。   size为0，capacity不变，内存不会释放

遍历
^^^^^^^^

.. code-block:: cpp

   #include <vector>
   using std::vector;
   vector<int> vInts(10,9);
   // 第一种方式
   for(int i=0;i<vInts.size();i++){
       cout<<vInts[i]<<endl;
   }
   // 第二种方式，迭代器
   for(vector<int>::iterator iter = vInts.begin(); iter != vInts.end(); iter++){
       cout<<*iter<<endl;
   }
   // c++ 11
   for (auto &i : vInts)
   {
       cout << i<< endl;
   }

查找
^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<vector>
   #include<algorithm>
   using namespace std;
   int main(){
       vector<int> vInts(10,9);
       vInts.insert(vInts.begin()+3,6);
       vector<int>::iterator res = find(vInts.begin(),vInts.end(),6);                           
       if(res == vInts.end()){
           cout<<"not find\n";
       }else{
           cout<<"find "<<*res<<endl;
       }   
   }

排序
^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<vector>
   #include<algorithm>
   using namespace std;
   int main(){
       vector<int> vInts{1,3,2,5,4};
       vInts.insert(vInts.begin()+3,6);
       sort(vInts.begin(),vInts.end());  //从小到大
       sort(vInts.rbegin(),vInts.rend());  //从大到小
   }

拼接
^^^^^^^^

.. code-block:: cpp

   #include <iostream>
   #include <vector>
   #include <algorithm>
   using namespace std;
   void show(vector<int> const &input) {
      for (auto const& i: input) {
         std::cout << i << " ";
      }
   }
   int main() {
      vector<int> v1 = { 1, 2, 3 };
      vector<int> v2 = { 4, 5 };
      v2.insert(v2.begin(), v1.begin(), v1.end());
      cout<<"Resultant vector is:"<<endl;
      show(v2);
      return 0;
   }

.. code-block:: text

   Resultant vector is:
   1 2 3 4 5

求和
^^^^^^^^^

.. code-block:: text

   T accumulate( InputIt first, InputIt last, T init );
   T accumulate( InputIt first, InputIt last, T init,BinaryOperation op );
   accumulate默认返回的是int类型，操作符默认是‘+’;当sum溢出时，将init类型改为long，则返回long类型

.. code-block:: cpp

   #include <iostream>
   #include <vector>
   #include <numeric>
   #include <string>
   #include <functional>

   int main()
   {
       std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
       int sum = std::accumulate(v.begin(), v.end(), 0);
       int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
       auto dash_fold = [](std::string a, int b) {
                            return std::move(a) + '-' + std::to_string(b);
                        };
       std::string s = std::accumulate(std::next(v.begin()), v.end(),
                                       std::to_string(v[0]), // 用首元素开始
                                       dash_fold);
       // 使用逆向迭代器右折叠
       std::string rs = std::accumulate(std::next(v.rbegin()), v.rend(),
                                        std::to_string(v.back()), // 用首元素开始
                                        dash_fold);
       std::cout << "sum: " << sum << '\n'
                 << "product: " << product << '\n'
                 << "dash-separated string: " << s << '\n'
                 << "dash-separated string (right-folded): " << rs << '\n';
   }

   sum: 55
   product: 3628800
   dash-separated string: 1-2-3-4-5-6-7-8-9-10
   dash-separated string (right-folded): 10-9-8-7-6-5-4-3-2-1

最大、最小值
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include <algorithm>
   #include <iostream>
   #include <vector>
   #include <cmath>

   static bool abs_compare(int a, int b)
   {
       return (std::abs(a) < std::abs(b));
   }

   int main() {
       const auto v = { 3, 9, 1, 4, 2, 5, 9 };
       const auto [min, max] = std::minmax_element(begin(v), end(v));

       std::cout << "min = " << *min << ", max = " << *max << '\n';

       std::vector<int>::iterator result = std::min_element(v.begin(), v.end());
       std::cout << "min element at: " << std::distance(v.begin(), result);

       result = std::max_element(v.begin(), v.end());
       std::cout << "max element at: " << std::distance(v.begin(), result) << '\n';

       result = std::max_element(v.begin(), v.end(), abs_compare);
       std::cout << "max element (absolute) at: " << std::distance(v.begin(), result) << '\n';
   }

翻转
^^^^^^^^^

.. code-block:: cpp

   # include<algorithm>
   const auto v = { 3, 9, 1, 4, 2, 5, 9 };
   std::reverse(v.begin(),v.end());


array
---------------------
vector是变长数组，array是定长数组

创建
^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    // CWG 1270 修订版之前的 C++11 中需要双括号（修订版后的 C++11 以及 C++14 及更高版本中不需要）
    std::array<int, 3> a1{ {1, 2, 3} };
    std::array<int, 3> a2 = {1, 2, 3}; 
    std::array<std::string, 2> a3 = { std::string("a"), "b" };
    // C++ 17
    std::array a4{3.0, 1.0, 4.0};  // -> std::array<double, 3>

获取容器大小
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    std::array<int, 3> a2 = {1, 2, 3}; 
    a2.size();
    a2.empty();

访问数据
^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    std::array<int, 3> a2 = {1, 2, 3}; 
    int t = a2.at(1);
    t = a2[1];
    t = a2.front();
    t = a2.back();
    a2.fill(0); //用全0填充

遍历
^^^^^^^^^^^^^^^^
.. code-block:: cpp

    std::array<std::string, 2> a3 = { std::string("a"), "b" };
    for(const auto& s: a3)
        std::cout << s << ' ';

List
-------

list容器就是一个双向链表,可以高效地进行插入删除元素

注意：list的iterator是双向的，只支持++、--。如果要移动多个元素应该用next：

https://www.cnblogs.com/scandy-yuan/archive/2013/01/08/2851324.html

2.1 创建
^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<list>
   using namespace std;
   int main(){
       list<int> c0; //空链表
       list<int> c1(3);  //建一个含三个默认值是0的元素的链表
       list<int> c2(5,2);  //建一个含五个元素的链表，值都是2
       list<int> c4(c2); //建一个c2的copy链表
       list<int> c5(c1.begin(),c1.end()); //c5含c1一个区域的元素[_First, _Last)  
       list<int> a1 {1,2,3,4,5};                                                             
       return 0;
   }

增加/插入数据
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<list>
   using namespace std;
   int main(){
       list<int> a{1,2,3,4,5},a1;
       a1 = a;
       a1.assign(5,10);  //assign(n,num)      将n个num拷贝赋值给链表c。
       list<int>::iterator it;
       for(it = a1.begin();it!=a1.end();it++){
           cout << *it << "\t";
           // 10      10      10      10      10
       }
       cout<<endl;
       a1.assign(a.begin(),a.end());   //assign(beg,end) 将[beg,end)区间的元素拷贝赋值给链表c。
       for(it = a1.begin();it!=a1.end();it++){
           cout << *it << "\t";
           // 1       2       3       4       5
       }
       cout<<endl;
       a1.insert(a1.begin(),0);  //insert(pos,num) 在pos位置插入元素num。返回插入元素对应的迭代器
       a1.insert(a1.begin(),2,88);  //insert(pos,n,num)      在pos位置插入n个元素num。
       int arr[5] = {11,22,33,44,55};
       a1.insert(a1.begin(),arr,arr+3);  //insert(pos,beg,end)      在pos位置插入区间为[beg,end)的元素。
       a1.insert(a1.begin(),a.begin(),a.end());
       a1.push_front(9);  //push_front(num)      在开始位置增加一个元素。
       a1.push_back(99);  //push_back(num)      在末尾增加一个元素。
       return 0;
   }

获取/修改容器大小
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   //c.empty(); // 判断链表是否为空。
   //c.size();  //返回链表c中实际元素的个数。
   //c.max_size(); //返回链表c可能容纳的最大元素数量。
   //resize(n)      从新定义链表的长度,超出原始长度部分用0代替,小于原始部分删除。
   //resize(n,num)            从新定义链表的长度,超出原始长度部分用num代替。
   #include<iostream>
   #include<list>
   using namespace std;
   int main(){
       list<int> a{1,2,3,4,5},a1;
       cout<<a.empty()<<";"<<a.size()<<";"<<a.max_size()<<endl;
       return 0;
   }

访问元素
^^^^^^^^^^^^

.. code-block:: cpp

   // c.front()      返回链表c的第一个元素。
   // c.back()      返回链表c的最后一个元素。
   #include <iterator>
   #include<list>
   using namespace std;
   int main(){
       list<int> a1{1,2,3,4,5};
       list<int>::iterator it;
       it = next(a1.begin(),3);
       iter = std::prev(it); //获取前一个迭代器
       cout<<*it<<endl;
       a1.clear();
       return 0;
   }

删除数据
^^^^^^^^^^^^

.. code-block:: cpp

   //c.clear();      清除链表c中的所有元素。
   //c.erase(pos)　　　　删除pos位置的元素。
   //c.pop_back()      删除末尾的元素。
   //c.pop_front()      删除第一个元素。
   //remove(num)             删除链表中匹配num的元素。
   #include<iostream>
   #include<list>
   #include <iterator>
   using namespace std;
   int main(){
       list<int> a1{1,2,3,4,5};
       list<int>::iterator it;
       a1.erase(next(a1.begin(),3));
       a1.pop_front();
       a1.pop_back();

       for(it = a1.begin();it!=a1.end();it++){
           cout << *it << "\t";
       }
       cout<<endl;
       a1.clear();
       return 0;
   }

遍历
^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<list>
   using namespace std;
   int main(){
       list<int> a1 {1,2,3,4,5};
       //正向遍历
       list<int>::iterator it;
       for(it = a1.begin();it!=a1.end();it++){
           cout << *it << "\t";
       }
       cout<<endl;
       //反向遍历
       list<int>::reverse_iterator itr;
       for(itr = a1.rbegin();itr!=a1.rend();itr++){
           cout << *itr << "\t";
       }
       cout<<endl;
       return 0;
   }

查找
^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<list>
   #include<algorithm>
   using namespace std;
   int main(){
       list<int> a1 {1,2,3,4,5};
       list<int>::iterator res = find(a1.begin(),a1.end(),3);                           
       if(res == a1.end()){
           cout<<"not find\n";
       }else{
           cout<<"find "<<*res<<endl;
       }   
   }

翻转
^^^^^^^^

.. code-block:: cpp

   //reverse()       反转链表
   list<int> a1{1,2,3,4,5};
   a1.reverse();

排序
^^^^^^^^

.. code-block:: cpp

   // c.sort()       将链表排序，默认升序
   // c.sort(comp)       自定义回调函数实现自定义排序
   #include<iostream>
   #include<list>
   #include <iterator>
   using namespace std;
   int main(){
       list<int> a1{1,3,2,5,4};
       a1.sort();
       a1.sort([](int n1,int n2){return n1>n2;});
       list<int>::iterator it;
       for(it = a1.begin();it!=a1.end();it++){
           cout << *it << "\t";
       }
       cout<<endl;
       return 0;
   }

去重
^^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<list>
   #include <iterator>
   using namespace std;
   int main(){
       list<int> a1{1,1,2,2,3,4,5};
       a1.unique();     //去重
       list<int>::iterator it;
       for(it = a1.begin();it!=a1.end();it++){
           cout << *it << "\t";
       }
       cout<<endl;
       return 0;
   }

map
------

创建
^^^^^^^^

.. code-block:: cpp

   #include <map>
   map<int, string> mm;
   //初始化列表来指定 map 的初始值
   std::map<std::string, size_t> people{{"Ann", 25}, {"Bill", 46},{"Jack", 32},{"Jill", 32}};
   std::map<std::string,size_t> people{std::make_pair("Ann",25),std::make_pair("Bill", 46),std::make_pair("Jack", 32),std::make_pair("Jill", 32)};
   //移动和复制构造函数
   std::map<std::string, size_t> personnel {people};
   //用另一个容器的一段元素来创建一个 map
   std::map<std::string, size_t> personnel {std::begin(people),std::end(people)};

增加/插入数据
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   //第一种：用insert函数插入pair数据 ,如果key存在，插入失败
   //第二种：用insert函数插入value_type数据，如果key存在，插入失败
   //第三种：用数组方式插入数据，如果key存在，覆盖value
   #include<iostream>
   #include<map>
   using namespace std;
   int main(){
       map<int, string> mm;
       pair<map<int, string>::iterator, bool> Insert_Pair;
       Insert_Pair = mm.insert(pair<int,string>(0,"zero"));  //插入pair数据
       if(Insert_Pair.second == true)
           cout<<"Insert Successfully"<<endl;
       else
           cout<<"Insert Failure"<<endl;
       mm.insert(make_pair(1,"one"));        //插入pair数据
       mm.insert(map<int,string>::value_type(3,"three"));  //插入value_type数据
       mm[4] = "four";                  //数组方式插入数据
       map<int, string>::iterator iter;
       for(iter = mm.begin(); iter != mm.end(); iter++)
           cout<<iter->first<<' '<<iter->second<<endl;
       return 0;
   }

获取/修改容器大小
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include<iostream>
   #include<map>
   using namespace std;
   int main(){
       map<int, string> mm;
       pair<map<int, string>::iterator, bool> Insert_Pair;
       Insert_Pair = mm.insert(pair<int,string>(0,"zero"));
       if(Insert_Pair.second == true)
           cout<<"Insert Successfully"<<endl;
       else
           cout<<"Insert Failure"<<endl;
       mm.insert(make_pair(1,"one"));
       mm.insert(map<int,string>::value_type(3,"three"));
       mm[4] = "four";
       int size = mm.size();  //获取map大小
       return 0;
   }

访问元素
^^^^^^^^^^^^

删除元素
^^^^^^^^^^^^

.. code-block:: cpp

   //iterator erase（iterator it);//通过一个条目对象删除
   //iterator erase（iterator first，iterator last）//删除一个范围
   //size_type erase(const Key&key);//通过关键字删除
   //clear()就相当于enumMap.erase(enumMap.begin(),enumMap.end());
   #include<iostream>
   #include<map>
   using namespace std;
   int main(){
       map<int, string> mm;
       mm.insert(pair<int,string>(0,"zero"));
       mm.insert(make_pair(1,"one"));
       mm.insert(map<int,string>::value_type(3,"three"));
       mm[4] = "four";
       map<int, string>::iterator iter;
       iter = mm.find(3);
       mm.erase(iter);    //迭代器删除
       int n = mm.erase(0);  //关键字删除，成功返回1，失败返回0
       for(iter = mm.begin(); iter != mm.end(); iter++)
           cout<<iter->first<<' '<<iter->second<<endl;
       mm.erase(mm.begin(),mm.end()); //全部删除
       return 0;
   }

遍历
^^^^^^^^

.. code-block:: cpp

   //第一种：应用前向迭代器
   //第二种：应用反相迭代器
   #include<iostream>
   #include<map>
   using namespace std;
   int main(){
       map<int, string> mm;
       mm.insert(pair<int,string>(0,"zero"));  //插入pair数据
       mm.insert(make_pair(1,"one"));        //插入pair数据
       mm.insert(map<int,string>::value_type(3,"three"));  //插入value_type数据
       mm[4] = "four";                  //数组方式插入数据
       map<int, string>::iterator iter;
       for(iter = mm.begin(); iter != mm.end(); iter++)
           cout<<iter->first<<' '<<iter->second<<endl;
       map<int, string>::reverse_iterator riter;  
       for(riter = mapStudent.rbegin(); riter != mapStudent.rend(); riter++)  
           cout<<riter->first<<"  "<<riter->second<<endl; 
       return 0;
   }

查找
^^^^^^^^

.. code-block:: cpp

   // 第一种：用count函数来判定关键字是否出现，其缺点是无法定位数据出现位置
   // 第二种：用find函数来定位数据出现位置，它返回的一个迭代器，当数据出现时，它返回数据所在位置的迭代器，如果map中没有要查找的数据，它返回的迭代器等于end函数返回的迭代器
   #include<iostream>
   #include<map>
   using namespace std;
   int main(){
       map<int, string> mm;
       mm.insert(pair<int,string>(0,"zero"));
       mm.insert(make_pair(1,"one"));
       mm.insert(map<int,string>::value_type(3,"three"));
       mm[4] = "four";
       map<int, string>::iterator iter;
       iter = mm.find(4);
       if(iter != mm.end()){
           cout<<"find key:"<<iter->first<<" value:"<<iter->second<<endl;
       }else{
           cout<<"not find"<<endl;
       }
       for(iter = mm.begin(); iter != mm.end(); iter++)
           cout<<iter->first<<' '<<iter->second<<endl;
       return 0;
   }

排序
^^^^^^^^

map中的元素是自动按Key升序排序，所以不能对map用sort函数,如果要是的key降序，使用：

.. code-block:: cpp

   std::map<int, int, std::greater<int>> mi;

STL中默认是采用小于号来排序的，以上代码在排序上是不存在任何问题的，因为上面的关键字是int 型，它本身支持小于号运算，在一些特殊情况，比如关键字是一个结构体，涉及到排序就会出现问题，因为它没有小于号操作，insert等函数在编译的时候过 不去；需要重载小于号

unordered_map
----------------

https://www.cnblogs.com/langyao/p/8823092.html

C++ 11标准中加入了unordered系列的容器。unordered_map记录元素的hash值，根据hash值判断元素是否相同,即unordered_map内部元素是无序的。

map中的元素是按照二叉搜索树存储（用红黑树实现），进行中序遍历会得到有序遍历。所以使用时map的key需要定义operator<

而unordered_map需要定义hash_value函数并且重载operator==

unordered_map编译时gxx需要添加编译选项：--std=c++11

queue
--------

创建
^^^^^^^^

.. code-block:: cpp

   queue<int> mqueue;
   queue<int> mqueue1{mqueue};

增加/插入数据
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   queue<int> mqueue;
   mqueue.push(1);
   mqueue.emplace(2);  //可以避免对象的拷贝，重复调用构造函数

获取/修改容器大小
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   queue<int> mqueue;
   mqueue.push(1);
   mqueue.emplace(2);
   mqueue.size();
   mqueue.empty();  //判断是否为空

访问元素
^^^^^^^^^^^^

.. code-block:: cpp

   mqueue.front();  //返回 queue 中第一个元素的引用
   mqueue.back();  //返回 queue 中最后一个元素的引用

删除元素
^^^^^^^^^^^^

.. code-block:: cpp

   mqueue.pop();

遍历
^^^^^^^^

和 stack 一样，queue 也没有迭代器。访问元素的唯一方式是遍历容器内容，并移除访问过的每一个元素

查找
^^^^^^^^

deque
--------

deque两端都能够快速插入和删除元素

Deque的操作函数和vector操作函数基本一模一样,duque的各项操作只有以下几点和vector不同:


#. deque不提供容量操作( capacity()、reserve() )
#. deque提供push_front()、pop_front()函数直接操作头部

deque元素是分布在一段段连续空间上，因此deque具有如下特点：

1、支持随机访问，即支持[]以及at()，但是性能没有vector好。

2、可以在内部进行插入和删除操作，但性能不及list。

 由于deque在性能上并不是最高效的，有时候对deque元素进行排序，更高效的做法是，将deque的元素移到到vector再进行排序，然后在移到回来。

创建
^^^^^^^^

.. code-block:: cpp

   deque<int> mqueue;
   deque<int>  d(10);  //创建容量为10的deque
   deque<int>  d2(6,8); //容量为6，所有元素初始化为8
   int ar[5]={1,2,3,4,5};   //使用数组的一个区间初始化
   deque<int>  d(ar,ar+5);
   vector<double> vd{0.1,0.2,.05,.07,0.9};  //使用vector的一个区间初始化
   deque<double>  d2(vd.begin()+1,vd.end());
   deque<int> mqueue1{mqueue};  //使用另一个deque初始化
   deque<int>  d2({1,2,3,4,5,6,7});  //初始化列表进行初始化

增加/插入数据
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   deque<int> mqueue;
   mqueue.push(1);
   mqueue.emplace_front(2);  //可以避免对象的拷贝，重复调用构造函数
   mqueue.emplace_back(2);  //可以避免对象的拷贝，重复调用构造函数

获取/修改容器大小
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   deque<int> mqueue;
   mqueue.push(1);
   mqueue.emplace_front(2);
   mqueue.size();
   mqueue.empty();  //判断是否为空

访问元素
^^^^^^^^^^^^

.. code-block:: cpp

   mqueue.front();  //返回 queue 中第一个元素的引用
   mqueue.back();  //返回 queue 中最后一个元素的引用

删除元素
^^^^^^^^^^^^

.. code-block:: cpp

   mqueue.pop_front();
   mqueue.pop_end();

遍历
^^^^^^^^

.. code-block:: cpp

   for (std::deque<int>::iterator it = dq.begin(); it!=dq.end(); ++it)
       std::cout << ' ' << *it;

查找
^^^^^^^^

stack
--------

创建
^^^^^^^^^

.. code-block:: cpp

   //stack<int> s1 = {1,2,3,4,5};   //error    stack不可以用一组数直接初始化
   //stack<int> s2(10);             //error    stack不可以预先分配空间
   stack<int> s3;

   vector<int> v1 = {1,2,3,4,5};       // 1,2,3,4,5依此入栈
   stack<int, vector<int>> s4(v1);

   list<int> l1 = {1,2,3,4,5};
   stack<int, list<int>> s5(l1);

   deque<int> d1 = {1,2,3,4,5};
   stack<int, deque<int>> s6(d1);
   stack<int> s7(d1);                  //用deque 为 stack  初始化时 deque可省  因为stack是基于deque, 默认以deque方式构造

增加/插入数据
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   mstack.push(333);
   mstach.emplace(333);

获取/修改容器大小
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   mstack.size();
   mstack.empty();

访问元素
^^^^^^^^^^^^

.. code-block:: cpp

   mstack.top();

删除元素
^^^^^^^^^^^^

.. code-block:: cpp

   mstack.pop();

遍历
^^^^^^^^

stack 遍历需要将所有元素出栈

.. code-block:: cpp

   #include<iostream>
   #include<stack>
   #include<deque>
   using namespace std;
   int main(){
       deque<int> q1{1,2,3,4,5};
       stack<int> s(q1);
       while(!s.empty()){
           cout<<s.top()<<" ";
           s.pop();
       }
       cout<<endl;
       return 0;
   }

priority_queue（堆）
-----------------------

和\ ``queue``\ 不同的就在于我们可以自定义其中数据的优先级, 让优先级高的排在队列前面,优先出队

优先队列具有队列的所有特性，包括基本操作，只是在这基础上添加了内部的一个排序，它本质是一个\ **二叉堆**\ 实现的

创建
^^^^^^^^

.. code-block:: cpp

   // 定义 priority_queue<Type, Container, Functional>
   // Type 就是数据类型，Container 就是容器类型（Container必须是用数组实现的容器，比如vector,deque等等，但不能用 list。STL里面默认用的是vector），
   // Functional 就是比较的方式，当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，
   // 默认是大顶堆

   //升序队列;小顶堆
   priority_queue <int,vector<int>,greater<int> > q;
   //降序队列；大顶堆
   priority_queue <int,vector<int>,less<int> >q;

.. code-block:: cpp

   //pari的比较，先比较第一个元素，第一个相等比较第二个
   #include <iostream>
   #include <queue>
   #include <vector>
   using namespace std;
   int main() 
   {
       priority_queue<pair<int, int> > a;
       pair<int, int> b(1, 2);
       pair<int, int> c(1, 3);
       pair<int, int> d(2, 5);
       a.push(d);
       a.push(c);
       a.push(b);
       while (!a.empty()) 
       {
           cout << a.top().first << ' ' << a.top().second << '\n';
           a.pop();
       }
   }

.. code-block:: cpp

   //自定义类型
   #include <iostream>
   #include <queue>
   using namespace std;

   //方法1
   struct tmp1 //运算符重载<
   {
       int x;
       tmp1(int a) {x = a;}
       bool operator<(const tmp1& a) const
       {
           return x < a.x; //大顶堆
       }
   };

   //方法2
   struct tmp2 //重写仿函数
   {
       bool operator() (tmp1 a, tmp1 b) 
       {
           return a.x < b.x; //大顶堆
       }
   };

   int main() 
   {
       tmp1 a(1);
       tmp1 b(2);
       tmp1 c(3);
       priority_queue<tmp1> d;
       d.push(b);
       d.push(c);
       d.push(a);
       while (!d.empty()) 
       {
           cout << d.top().x << '\n';
           d.pop();
       }
       cout << endl;

       priority_queue<tmp1, vector<tmp1>, tmp2> f;
       f.push(c);
       f.push(b);
       f.push(a);
       while (!f.empty()) 
       {
           cout << f.top().x << '\n';
           f.pop();
       }
   }

增加/插入数据
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   priority_queue<int> mqueue;
   mqueue.push(1);
   mqueue.emplace(2);  //可以避免对象的拷贝，重复调用构造函数

获取/修改容器大小
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   priority_queue<int> mqueue;
   mqueue.push(1);
   mqueue.emplace(2);
   mqueue.size();
   mqueue.empty();  //判断是否为空
   mqueue.clear();  // 清空所有元素

访问元素
^^^^^^^^^^^^

.. code-block:: cpp

   mqueue.top();  //返回 queue中第一个元素，即最大/最小的元素

删除元素
^^^^^^^^^^^^

.. code-block:: cpp

   mqueue.pop();

遍历
^^^^^^^^

和 stack 一样，queue 也没有迭代器。访问元素的唯一方式是遍历容器内容，并移除访问过的每一个元素

排列组合
-----------

**next_permutation和prev_permutation区别：**

next_permutation（start,end），和prev_permutation（start,end）。这两个函数作用是一样的，区别就在于前者求的是当前排列的下一个排列，后一个求的是当前排列的上一个排列。至于这里的“前一个”和“后一个”，我们可以把它理解为序列的字典序的前后，严格来讲，就是对于当前序列pn，他的下一个序列pn+1满足：不存在另外的序列pm，使pn<pm<pn+1.

生成N个不同元素的全排列
^^^^^^^^^^^^^^^^^^^^^^^^^^^

这是next_permutation()的基本用法，把元素从小到大放好（即字典序的最小的排列），然后反复调用next_permutation()就行了

.. code-block:: cpp

   #include<iostream>
   #include <iterator>
   #include<string>
   #include <vector>
   #include <algorithm>

   int main(int argc, char *argv[]) {
     std::vector<int> vec{1,2,3,4};
     int count=0;
     do{
       std::cout<<++count<<":";
       std::copy(vec.begin(),vec.end(),std::ostream_iterator<int>(std::cout,","));
       std::cout<<std::endl;
     }while(std::next_permutation(vec.begin(),vec.end()));
   }

带有重复字符的排列组合

.. code-block:: cpp

   #include <algorithm>
   #include <string>
   #include <iostream>

   int main()
   {
       std::string s = "aba";
       std::sort(s.begin(), s.end());
       do {
           std::cout << s << '\n';
       } while(std::next_permutation(s.begin(), s.end()));
   }

生成从N个元素中取出M个的所有组合
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**题目：**\ 输出从7个不同元素中取出3个元素的所有组合

思路：对序列{1,1,1,0,0,0,0}做全排列。对于每个排列，输出数字1对应的位置上的元素。

.. code-block:: cpp

   #include<iostream>
   #include <iterator>
   #include<string>
   #include <vector>
   #include <algorithm>

   int main(int argc, char *argv[]) {


     std::vector<int> values{1,2,3,4,5,6,7};
     std::vector<int> selectors{1,1,1,0,0,0,0};
     int count=0;
     do{
       std::cout<<++count<<": ";
       for(size_t i=0;i<selectors.size();i++){
         if(selectors[i]){
           std::cout<<values[i]<<", ";
         }
       }
       std::cout<<std::endl;
     }while(std::prev_permutation(selectors.begin(),selectors.end()));
   }


unique(去重)
----------------

std::unique()的作用是去除相邻的重复元素，可以自定义判断元素重复的方法

.. code-block:: cpp

   #include<iostream>
   #include <iterator>
   #include<string>
   #include <vector>
   #include <algorithm>

   bool bothSpaces(char x,char y){
     return x==' ' && y== ' ';
   }

   int main(int argc, char *argv[]) {
     std::string str = "abcc     aab            c";
     std::string str1 = str;
     std::string::iterator last = std::unique(str.begin(),str.end());
     str.erase(last,str.end());  
     std::cout<<str<<std::endl;  //abc ab c

     std::string::iterator last1 = std::unique(str1.begin(),str1.end(),bothSpaces);
     str1.erase(last1,str1.end());
     std::cout<<str1<<std::endl;  //abcc aab c
   }

std::unique()通用适用于容器；

**注意：**\ unique之后， 容器元素被修改了，但是个数没变，需要手动调整容器的大小，这个位置由unique的返回值来确定

.. code-block:: cpp

   #include<iostream>
   #include <iterator>
   #include<string>
   #include <vector>
   #include <algorithm>

   int main(int argc, char *argv[]) {
     std::vector<int> vi{1,2,2,3,2,1,1};
     auto result = unique(vi.begin(), vi.end());
     vi.resize(std::distance(vi.begin(), result));
     std::copy(vi.begin(), vi.end(), std::ostream_iterator<int>(std::cout, ","));
     return 0;
   }

set
-------

set是一种关联\ `容器 <https://www.geeksforgeeks.org/containers-cpp-stl/>`_\ ，其中每个元素都必须是唯一的，这些值按特定顺序存储。

底层实现是平衡二叉查找树，典型的用法不是使用AVL树，而是使用自顶向下的红黑树。

特性：


#. set中存储的值是排序的（如果要用乱序的，使用unordered_set）
#. set中的值是唯一的
#. 加入到set中的值不可改变；要改变需要删除原有值，添加新值
#. set底层是基于二叉搜索树实现的
#. set集合中的值不可以通过下标索引

默认情况下，排序操作使用less<Object>函数对象实现，该函数对象是通过对Object调用operator<来实现的。
另一种可替代的排序方案可以通过具有函数对象类型的set模板来实现。
例如，可以生成一个存储string对象的set，通过使用CaseInsensitiveCompare函数对象来忽略字符的大小写。

.. code-block:: cpp

    std::set<std::string,CaseInsensitiveCompare> s;

创建
^^^^^^^^^

.. code-block:: cpp

   set<int> val; //创建一个空的set
   set<int> val = {6, 10, 5, 1}; // 使用值初始化set
   set<int, greater<int> > s1;  // 创建一个空的set，自定义排序方法
   set<int> s2(s1.begin(), s1.end());  // 从其他set集合中拷贝

增加/插入数据
^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   // 返回插入元素所在位置的迭代器
   iterator set_name.insert(element)

.. code-block:: cpp

   #include <bits/stdc++.h>
   using namespace std;
   int main()
   {
       set<int> s;
       // Function to insert elements
       // in the set container
       s.insert(1);
       s.insert(4);
       s.insert(2);
       s.insert(5);
       s.insert(3);
       cout << "The elements in set are: ";
       for (auto it = s.begin(); it != s.end(); it++)
           cout << *it << " ";

       return 0;
   }

获取/修改容器大小
^^^^^^^^^^^^^^^^^^^^^^

只能获取set的大小，不能直接修改set的大小

.. code-block:: cpp

   #include <bits/stdc++.h>
   using namespace std;
   int main()
   {
       set<int> s;
       // Function to insert elements
       // in the set container
       s.insert(1);
       s.insert(4);
       s.insert(2);
       s.insert(5);
       s.insert(3);
       cout << "The elements in set size: "<<s.size();
       return 0;
   }

访问元素
^^^^^^^^^^^^^

set只能通过迭代器访问

删除元素
^^^^^^^^^^^^^

.. code-block:: cpp

   #include <bits/stdc++.h>
   using namespace std;
   int main()
   {
       set<int> s = {1,4,2,5,3};
       cout << "The elements in set are: ";
       for (auto it = s.begin(); it != s.end(); it++)
           cout << *it << " ";
       s.erase(s.begin(), s.find(3)); //删除小于3的所有元素
       s.erase(5); // 删除指定元素
       return 0;
   }

遍历
^^^^^^^^^

.. code-block:: cpp

   #include <bits/stdc++.h>
   using namespace std;
   int main()
   {
       set<int> s = {1,4,2,5,3};
       cout << "The elements in set are: ";
       for (auto it = s.begin(); it != s.end(); it++)
           cout << *it << " ";
       return 0;
   }

查找
^^^^^^^^^

.. code-block:: cpp

   #include <bits/stdc++.h>
   using namespace std;
   int main()
   {
       // Initialize set
       set<int> s;
       s.insert(1);
       s.insert(4);
       s.insert(2);
       s.insert(5);
       s.insert(3);
       // iterator pointing to
       // position where 3 is
       auto pos = s.find(3);
       // prints the set elements
       cout << "The set elements after 3 are: ";
       for (auto it = pos; it != s.end(); it++)
           cout << *it << " ";
       return 0;
   }

hash
--------

哈希模板定义一个函数对象，实现了\ `散列函数 <http://en.wikipedia.com/wiki/Hash_function>`_\ 。这个函数对象的实例定义一个operator()


#. 接受一个参数的类型\ ``Key``.
#. 返回一个类型为size_t的值，表示该参数的哈希值.
#. 调用时不会抛出异常.
#. 若两个参数`k1` `k2` 相等，则std::hash<Key>()(k1)== std::hash<Key>()(k2).
#. 若两个不同的参数 `k1` `k2`不相等，则std::hash<Key>()(k1)== std::hash<Key>()(k2)成立的概率应非常小，接近1.0/\ `std::numeric_limits <http://zh.cppreference.com/w/cpp/types/numeric_limits>`_ <size_t>::max().

.. code-block:: cpp

   #include <iostream>
   #include <iomanip>
   #include <functional>
   #include <string>
   #include <unordered_set>

   struct S {
       std::string first_name;
       std::string last_name;
   };
   bool operator==(const S& lhs, const S& rhs) {
       return lhs.first_name == rhs.first_name && lhs.last_name == rhs.last_name;
   }

   // 自定义散列函数能是独立函数对象：
   struct MyHash
   {
       std::size_t operator()(S const& s) const 
       {
           std::size_t h1 = std::hash<std::string>{}(s.first_name);
           std::size_t h2 = std::hash<std::string>{}(s.last_name);
           return h1 ^ (h2 << 1); // 或使用 boost::hash_combine （见讨论）
       }
   };

   // std::hash 的自定义特化能注入 namespace std
   namespace std
   {
       template<> struct hash<S>
       {
           typedef S argument_type;
           typedef std::size_t result_type;
           result_type operator()(argument_type const& s) const
           {
               result_type const h1 ( std::hash<std::string>{}(s.first_name) );
               result_type const h2 ( std::hash<std::string>{}(s.last_name) );
               return h1 ^ (h2 << 1); // 或使用 boost::hash_combine （见讨论）
           }
       };
   }

   int main()
   {

       std::string str = "Meet the new boss...";
       std::size_t str_hash = std::hash<std::string>{}(str);
       std::cout << "hash(" << std::quoted(str) << ") = " << str_hash << '\n';

       S obj = { "Hubert", "Farnsworth"};
       // 使用独立的函数对象
       std::cout << "hash(" << std::quoted(obj.first_name) << ',' 
                  << std::quoted(obj.last_name) << ") = "
                  << MyHash{}(obj) << " (using MyHash)\n                           or "
                  << std::hash<S>{}(obj) << " (using std::hash) " << '\n';

       // 自定义散列函数令在无序容器中使用自定义类型可行
       // 此示例将使用注入的 std::hash 特化，
       // 若要使用 MyHash 替代，则将其作为第二模板参数传递
       std::unordered_set<S> names = {obj, {"Bender", "Rodriguez"}, {"Leela", "Turanga"} };
       for(auto& s: names)
           std::cout << std::quoted(s.first_name) << ' ' << std::quoted(s.last_name) << '\n';
   }


std::copy
-----------------
拷贝[first,last)区间的数据到result

.. code-block:: cpp

    template<class InputIterator, class OutputIterator>
    OutputIterator copy (InputIterator first, InputIterator last, OutputIterator result)
    {
    while (first!=last) {
        *result = *first;
        ++result; ++first;
    }
    return result;
    }

.. code-block:: cpp

    #include <iostream>     // std::cout
    #include <algorithm>    // std::copy
    #include <vector>       // std::vector
    int main () {
        int myints[]={10,20,30,40,50,60,70};
        std::vector<int> myvector (7);

        std::copy ( myints, myints+7, myvector.begin() );
        std::cout << "myvector contains:";
        for (std::vector<int>::iterator it = myvector.begin(); it!=myvector.end(); ++it)
            std::cout << ' ' << *it;
        std::cout << '\n';
        return 0;
    }

std::copy_n
-------------------
拷贝first开始的n个数据到result

.. code-block:: cpp

    template< class InputIt, class Size, class OutputIt>
    OutputIt copy_n(InputIt first, Size count, OutputIt result)

.. code-block:: cpp

    #include <algorithm>
    int main()
    {
        std::string in = "1234567890";
        std::string out;
    
        std::copy_n(in.begin(), 4, std::back_inserter(out));
        std::cout << out << '\n';
    
        std::vector<int> v_in(128);
        std::iota(v_in.begin(), v_in.end(), 1);
        std::vector<int> v_out(v_in.size());
    
        std::copy_n(v_in.cbegin(), 100, v_out.begin());
        std::cout << std::accumulate(v_out.begin(), v_out.end(), 0) << '\n';
    }

.. _std::ref:

std::ref 
--------------
C++11 中引入 std::ref 用于取某个变量的引用，这个引入是为了解决一些传参问题。

std::bind，std::thread 必须显式通过 std::ref 来绑定引用进行传参，否则，形参的引用声明是无效的

std::string_view
--------------------------
C++17中我们可以使用std::string_view来获取一个字符串的视图，字符串视图并不真正的创建或者拷贝字符串，
而只是拥有一个字符串的查看功能。std::string_view比std::string的性能要高很多，
因为每个std::string都独自拥有一份字符串的拷贝，而std::string_view只是记录了自己对应的字符串的指针和偏移位置。
当我们在只是查看字符串的函数中可以直接使用std::string_view来代替std::string。

注意：因为std::string_view是原始字符串的视图，如果在查看std::string_view的同时修改了字符串，或者字符串被消毁，那么将是未定义的行为。

.. code-block:: cpp

    #include <string_view>
    int main(int argc, char *argv[]) {
        const char *cstr = "qwertyuiolkjhg";
        std::string_view stringView1(cstr);
        std::string_view stringView2(cstr, 4);
        std::cout << "stringView1: " << stringView1 << ", stringView2: " << stringView2 << std::endl;

        std::string str = "qwertyuiolkjhg";
        std::string_view stringView3(str.c_str());
        std::string_view stringView4(str.c_str(), 4);
        std::cout << "stringView3: " << stringView1 << ", stringView4: " << stringView2 << std::endl;
        return 0;
    }


.. _std::move:

std::move
----------------------
* 不是移动：move函数并不做具体移动操作，其目的只是高数编译器当前对象具备可移动条件
* 类型转换：本质一种前置类型转换，将参数转换为右值，可以理解为“右值类型转换”(rvalue_cast)编译时特征，对运行期无影响
* 不保证：并不必然导致移动构造或赋值发生，还要看参数是否符合其他条件

    - 如果参数本身不支持移动构造和赋值
    - 如果是对const左值参数使用std::move,移动不接受常量性参数

* 退化拷贝：如果不能满足移动的条件，对移动的请求最后还会退回拷贝操作

.. _std::forward:

std::forward
------------------------
* 应用于转发引用
* 有条件的编译时类型转换，没有任何运行时计算

    - 当传入的参数是右值，forward将类似std::move函数，转换为右值，从而保留参数的右值特性
    - 当传入的参数是左值，forward将什么都不做，继续保留参数的左值特性

* 不要对转发引用调用std::move，因为可能是左值
* 如果没有forward，很多函数需要同时提供两种重载(传入左值时，使用左值引用；传入右值是，使用右值引用)，代码重复且易错

std::bind
-------------------------
std::bind的头文件是 <functional>，它是一个函数适配器，接受一个可调用对象（callable object），
生成一个新的可调用对象来“适应”原对象的参数列表。

std::bind将可调用对象与其参数一起进行绑定，绑定后的结果可以使用std::function保存。std::bind主要有以下两个作用：
* 将可调用对象和其参数绑定成一个仿函数；
* 只绑定部分参数，减少可调用对象传入的参数。

绑定普通函数
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    void print1(int data, string prefix )
    {
        cout<<prefix << data ;
    }
    //返回一个函数对象（类型为std::function)
    // _1表示占位符，位于<functional>中，std::placeholders::_1；
    auto binder=std::bind(print1, _1, " * ");
    binder(100); // print1(100,"*");

绑定一个成员函数
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    class Base
    {
        void display_sum(int a1, int a2)
        {
            std::cout << a1 + a2 << '\n';
        }
        int m_data = 30;
    };
    int main() 
    {
        Base base;
        auto newiFunc = std::bind(&Base::display_sum, &base, 100, std::placeholders::_1);
        f(20); // should out put 120. 
    }

绑定一个引用参数
^^^^^^^^^^^^^^^^^^^^^^
默认情况下，bind的那些不是占位符的参数被拷贝到bind返回的可调用对象中。但是，与lambda类似，
有时对有些绑定的参数希望以引用的方式传递，或是要绑定参数的类型无法拷贝

.. code-block:: cpp

    #include <iostream>
    #include <functional>
    #include <vector>
    #include <algorithm>
    #include <sstream>
    using namespace std::placeholders;
    using namespace std;
    
    ostream & printInfo(ostream &os, const string& s, char c)
    {
        os << s << c;
        return os;
    }
    
    int main()
    {
        vector<string> words{"welcome", "to", "C++11"};
        ostringstream os;
        char c = ' ';
        for_each(words.begin(), words.end(), 
                    [&os, c](const string & s){os << s << c;} );
        cout << os.str() << endl;
        ostringstream os1;
        // ostream不能拷贝，若希望传递给bind一个对象，
        // 而不拷贝它，就必须使用标准库提供的ref函数
        for_each(words.begin(), words.end(),
                    bind(printInfo, ref(os1), _1, c));
        cout << os1.str() << endl;
    }

std::function
--------------------------
std::function是一个函数包装器模板，最早来自boost库，对应其boost::function函数包装器。在c++0x11中，将boost::function纳入标准库中。
该函数包装器模板能包装任何类型的可调用元素（callable element），例如普通函数和函数对象。
包装器对象可以进行拷贝，并且包装器类型仅仅只依赖于其调用特征（call signature），而不依赖于可调用元素自身的类型。

一个std::function类型对象实例可以包装下列这几种可调用元素类型：函数、函数指针、类成员函数指针或任意类型的
函数对象（例如定义了operator()操作并拥有函数闭包）。std::function对象可被拷贝和转移，并且可以使用指定的调用特征来直接调用目标元素。
当std::function对象未包裹任何实际的可调用元素，调用该std::function对象将抛出std::bad_function_call异常。

std::invoke
-----------------------------
万能调用

.. code-block:: cpp

    #include <functional>
    #include <iostream>
    
    struct Foo {
        Foo(int num) : num_(num) {}
        void print_add(int i) const { std::cout << num_+i << '\n'; }
        int num_;
    };
    
    void print_num(int i){
        std::cout << i << '\n';
    }
    
    struct PrintNum {
        void operator()(int i) const{
            std::cout << i << '\n';
        }
    };
    
    int main()
    {
        // 调用自由函数
        std::invoke(print_num, -9);
        // 调用 lambda
        std::invoke([]() { print_num(42); });
        // 调用成员函数
        const Foo foo(314159);
        std::invoke(&Foo::print_add, foo, 1);
        // 调用（访问）数据成员
        std::cout << "num_: " << std::invoke(&Foo::num_, foo) << '\n';
        // 调用函数对象
        std::invoke(PrintNum(), 18);
    }


std::pair
----------------------
* 标准库类型std::pair,将两个值组合在一个对象中
* std::pair是一个模板类，通过自动类型推导，来提供两个数据成员first和second的类型
* C++17之前，支持模板函数自动类型推导，即make_pair
* C++17之后，支持构造函数参数类型推导
* std::pair支持拷贝，移动，析构，是对first和second的委托调用
  

.. code-block:: cpp

    std::pair<int, double> p1{42,3.1415};
    cout<<p1.first<<","<<p1.second<<endl;
    auto p2=make_pair(42, 3.1415); //std::pair<int, double> p2{42, 3.1415};
    std::pair p3{42, 3.1415};  // C++17支持


std::tuple
----------------------
* 标准库类型std::tuple,包含N个任意类型元素的序列
* std::tuple是一个变参模板类，支持自动类型推导，通过get<>函数获取元素
* C++17之前，支持模板函数自动类型推导，即make_tuple
* C++17之后，支持构造函数参数类型推导
* std::tuple支持拷贝，移动，析构，是对其内各元素的委托调用

.. code-block:: cpp

    auto t = std::make_tuple(100, "C++ Programming Language", 100.5);
    cout  << std::get<0>(book1) << ", " << std::get<1>(book1) << std::get<2>(book1) << '\n';
    int id1;
    string name1;
    double price1;
    std::tie(id1, name1, price1) = t;
    cout<<id1<<", "<<name1<<", "<<price1<< '\n';
    auto [id2, name2, price2] = t;  //结构化绑定


std::optional
---------------------
* 标准库std::optional，通过一个内部bool标记，来表示可能包含值，也可能是空的类型
* std::optional常用语可空的返回值，参数，数据成员，使用make_optional或构造器创建
* std::optional内存结构：除了其内存存储的值对象size外，还会增加一个bool存储(无论是否有有效值)
* 可以使用操作符\*,->,value(),value_or()来访问有效值
* 支持拷贝，移动，析构，前提是存在有效值，是对其内有效值的委托

.. code-block:: cpp

    #include <string>
    #include <functional>
    #include <iostream>
    #include <optional>
    
    // optional can be used as the return type of a factory that may fail
    std::optional<std::string> create(bool b) {
        if (b)
            return "Godzilla";
        return {};
    }
    
    // std::nullopt can be used to create any (empty) std::optional
    auto create2(bool b) {
        return b ? std::optional<std::string>{"Godzilla"} : std::nullopt;
    }
    
    // std::reference_wrapper may be used to return a reference
    auto create_ref(bool b) {
        static std::string value = "Godzilla";
        return b ? std::optional<std::reference_wrapper<std::string>>{value}
                : std::nullopt;
    }
    
    int main()
    {
        std::cout << "create(false) returned "
                << create(false).value_or("empty") << '\n';
    
        // optional-returning factory functions are usable as conditions of while and if
        if (auto str = create2(true)) {
            std::cout << "create2(true) returned " << *str << '\n';
        }
    
        if (auto str = create_ref(true)) {
            // using get() to access the reference_wrapper's value
            std::cout << "create_ref(true) returned " << str->get() << '\n';
            str->get() = "Mothra";
            std::cout << "modifying it changed it to " << str->get() << '\n';
        }
    }

std::variant
----------------------
* 使用模板参数，存储多个强类型参数中的一个（存有当前类型的索引）
* union与std::variant
  
  - union类型不安全，不支持对象语义和RAII
  - variant类型安全，且支持对象语义和RAII
  - 基本类型使用union，自定义类型使用std::variant

* 使用std::visit搭配visitor，支持对类型的多态访问
* 内存模型：最大的类型+8个bytes的索引，无需额外的堆内存分配
* 使用std::variant的典型场合：

  - 返回值或参数
  - 特殊错误处理variant<ReturnObject,ErrorCode>
  - 不适用虚函数的多态visitor

.. code-block:: cpp

    using WidgetABC = std::variant<WidgetA, WidgetB,WidgetC>;
    WidgetABC w1=WidgetA{};


std::any
---------------------
* 可以存储任何支持拷贝构造的类型，包括基本类型和自定义类型
* 内部存储类型信息，类型安全，使用any_cast转型，违例抛异常
* 支持正常的构造，移动，赋值，析构
* 内存模型使用小对象优化
* 适用于“任意类型场合”：无类型容器，配置文件解析，与脚本语言交互
* 尽量避免使用std::any(特别是存储基本类型，有缓存折损)，如果有强类型，使用模板，如果有父类，使用继承层次

.. code-block:: cpp

    Widget w;
    std::any any1=100;
    std::any any2="hello"s;
    std::any any3=w;


STL中的排序算法
-------------------------
在STL中，排序算法是通过使用函数模板sort来完成的。sort的参数是容器的头尾标志以及一个可选的比较器

.. code-block:: cpp

    void sort(Iterator begin,Iterator end);
    void sort(Iterator begin,Iterator end,Comparator cmp);

    sort(v.begin(),v.end()); // 升序排列
    sort(v.begin(),v.end(),greater<int>()); // 降序排列
    sort(v.begin(),v.begin()+(v.end()-v.begin())/2); // 对前半部分升序排列

sort算法不能保证相等的项保持它们原始的序列(如果这很重要的话，可以使用stable_sort来替代sort)。


其他使用
--------

numeric_limits-获取最大最小值
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include <limits>
   double min_dist = numeric_limits<double>::max();
   double max_dist = numeric_limits<double>::min();
