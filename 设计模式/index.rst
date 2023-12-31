设计模式
==================================

.. toctree::
   :maxdepth: 1
   :caption: 设计模式
   :name: 设计模式

   工厂模式
   简单工厂模式
   工厂方法模式
   抽象工厂模式
   建造者模式
   单例模式
   适配器模式
   桥接模式
   装饰模式
   外观模式
   享元模式
   代理模式
   命令模式
   中介者模式
   观察者模式
   状态模式
   策略模式
   模板方法模式
   原型模式
   访问器模式
   组合模式
   迭代器模式
   职责链模式
   备忘录模式
   解释器模式
   注册器模式


创建型模式
----------------------
创建型模式(Creational Pattern)对类的实例化过程进行了抽象，能够将软件模块中对象的创建和对象的使用分离。
为了使软件的结构更加清晰，外界对于这些对象只需要知道它们共同的接口，而不清楚其具体的实现细节，使整个系统的设计更加符合单一职责原则。

创建型模式在创建什么(What)，由谁创建(Who)，何时创建(When)等方面都为软件设计者提供了尽可能大的灵活性。
创建型模式隐藏了类的实例的创建细节，通过隐藏对象如何被创建和组合在一起达到使整个系统独立的目的。

创建型模式包含：

* :ref:`简单工厂模式`  重要程度：4 Factory
* :ref:`工厂方法模式`  重要程度：5 Factory Method
* :ref:`抽象工厂模式`  重要程度：5 Abstract Factory
* :ref:`建造者模式`    重要程度：2 Builder
* :ref:`单例模式`      重要程度：4 Singleton
* :ref:`原型模式`  Prototype



结构型模式
--------------------
结构型模式(Structural Pattern)描述如何将类或者对 象结合在一起形成更大的结构，就像搭积木，
可以通过 简单积木的组合形成复杂的、功能更为强大的结构。

结构型模式可以分为类结构型模式和对象结构型模式：

* 类结构型模式关心类的组合，由多个类可以组合成一个更大的系统，在类结构型模式中一般只存在继承关系和实现关系。 
* 对象结构型模式关心类与对象的组合，通过关联关系使得在一 个类中定义另一个类的实例对象，然后通过该对象调用其方法。 
  根据“合成复用原则”，在系统中尽量使用关联关系来替代继 承关系，因此大部分结构型模式都是对象结构型模式。

* :ref:`适配器模式` 重要程度：4 Adapter
* :ref:`桥接模式`   重要程度：3 Bridge
* :ref:`装饰模式`   重要程度：3 Decorator
* :ref:`外观模式`   重要程度：5 Facade
* :ref:`享元模式`   重要程度：5 Flyweight
* :ref:`代理模式`   重要程度：5 Proxy
* :ref:`组合模式` Composite
* :ref:`注册器模式` 




行为型模式
------------------
行为型模式(Behavioral Pattern)是对在不同的对象之间划分责任和算法的抽象化。

行为型模式不仅仅关注类和对象的结构，而且重点关注它们之间的相互作用。

通过行为型模式，可以更加清晰地划分类与对象的职责，并研究系统在运行时实例对象之间的交互。
在系统运行时，对象并不是孤立的，它们可以通过相互通信与协作完成某些复杂功能，一个对象在运行时也将影响到其他对象的运行。

行为型模式分为 **类行为型模式** 和 **对象行为型模式** 两种：

* 类行为型模式：类的行为型模式使用继承关系在几个类之间分配行为，类行为型模式主要通过多态等方式来分配父类与子类的职责。
* 对象行为型模式：对象的行为型模式则使用对象的聚合关联关系来分配行为，对象行为型模式主要是通过对象关联等方式来分配两个或多个类的职责。
  根据“合成复用原则”，系统中要尽量使用关联关系来取代继承关系，因此大部分行为型设计模式都属于对象行为型设计模式。


行为模式包含：

* :ref:`命令模式` 重要程度：3 Commond
* :ref:`中介者模式` 重要程度：4  Mediator
* :ref:`观察者模式` 重要程度：5  Observer
* :ref:`状态模式` 重要程度：3  State
* :ref:`策略模式` 重要程度：4  Strategy
* :ref:`模板方法模式`  Template Method
* :ref:`迭代器模式`  Iterator
* :ref:`职责链模式`  Chain of Responsibility
* :ref:`解释器模式`  Interpreter
* :ref:`备忘录模式`  Memento
* :ref:`访问器模式`  Visitor


从管理变化的角度理解设计模式
--------------------------------------------------------
+----------+--------------------------------------------+
|   分类   |                    模式                    |
+==========+============================================+
| 晚期扩展 | :ref:`模板方法模式`  Template Method       |
+----------+--------------------------------------------+
|          | :ref:`建造者模式` Builder                  |
+----------+--------------------------------------------+
| 策略对象 | :ref:`策略模式` Strategy                   |
+----------+--------------------------------------------+
|          | :ref:`观察者模式`  Observer                |
+----------+--------------------------------------------+
| 对象创建 | :ref:`简单工厂模式`  Factory               |
+----------+--------------------------------------------+
|          | :ref:`工厂方法模式` Factory Method         |
+----------+--------------------------------------------+
|          | :ref:`抽象工厂模式` Abstract Factory       |
+----------+--------------------------------------------+
|          | :ref:`原型模式`  Prototype                 |
+----------+--------------------------------------------+
| 单一职责 | :ref:`桥接模式` Bridge                     |
+----------+--------------------------------------------+
|          | :ref:`装饰模式` Decorator                  |
+----------+--------------------------------------------+
| 行为变化 | :ref:`命令模式`  Commond                   |
+----------+--------------------------------------------+
|          | :ref:`访问器模式`  Visitor                 |
+----------+--------------------------------------------+
| 接口隔离 | :ref:`适配器模式` Adapter                  |
+----------+--------------------------------------------+
|          | :ref:`代理模式` Proxy                      |
+----------+--------------------------------------------+
|          | :ref:`外观模式` Facade                     |
+----------+--------------------------------------------+
|          | :ref:`中介者模式` Mediator                 |
+----------+--------------------------------------------+
| 对象性能 | :ref:`单例模式` Singleton                  |
+----------+--------------------------------------------+
|          | :ref:`享元模式` Flyweight                  |
+----------+--------------------------------------------+
| 数据结构 | :ref:`组合模式` Composite                  |
+----------+--------------------------------------------+
|          | :ref:`迭代器模式`  Iterator                |
+----------+--------------------------------------------+
|          | :ref:`职责链模式`  Chain of Responsibility |
+----------+--------------------------------------------+
| 状态变化 | :ref:`状态模式` State                      |
+----------+--------------------------------------------+
|          | :ref:`备忘录模式`  Memento                 |
+----------+--------------------------------------------+
| 领域规则 | :ref:`解释器模式` Interpreter              |
+----------+--------------------------------------------+


参考
----------------------
https://design-patterns.readthedocs.io/zh_CN/latest/behavioral_patterns/strategy.html

https://www.runoob.com/design-pattern/state-pattern.html 
   
   
