pdb调试python程序
====================================

``python -m pdb test.py``

会自动停在第一行，等待调试

* break 或 b	设置断点
* continue 或 c	继续执行程序
* list 或 l	查看当前行的代码段
* step 或 s	进入函数
* return 或 r	执行代码直到从当前函数返回
* exit 或 q	中止并退出
* next 或 n	执行下一行
* pp	打印变量的值
* help	帮助


.. code-block:: shell

    p dir(my_object)  # 打印对象属性
    p vars(my_object) # 打印对象属性和值
    p my_object.__dict__  # 打印对象属性和值