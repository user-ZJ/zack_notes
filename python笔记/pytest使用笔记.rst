pytest使用笔记
===========================

pytest安装
-----------------
.. code-block:: shell

    pip install pytest


第一个例子
------------------
.. code-block:: python 

    # content of test_sample.py
    def func(x):
        return x + 1

    def test_answer():
        assert func(3) == 5

.. code-block:: shell

    pytest

pytest将在当前目录及其子目录中运行 ​test_*.py​ 或 ​*_test.py​ 形式的所有文件。

按照其 Python 测试约定发现所有测试(找到所有以 ​test_ ​为前缀的函数)


可以使用 `pytest xxx.py` 运行某个文件中的测试用例

测试类
--------------
.. code-block:: python

    class TestClassDemoInstance:
        value = 0

        def test_one(self):
            self.value = 1
            assert self.value == 1

        def test_two(self):
            assert self.value == 1

* pytest会发现所有类内以 ​test_ ​为前缀的函数
* 在类中对测试进行分组时需要注意的是，每个测试都有一个唯一的类实例。 
* 使用 `pytest -k TestClassDemoInstance` 运行类内的所有测试用例

pytest调用
-------------------
.. code-block:: shell

    # 运行test_mod.py中所有用例
    pytest test_mod.py
    # 运行目录下所有用例
    pytest testing/
    # 通过关键字表达式运行测试
    # 运行包含与给定字符串表达式匹配的名称（不区分大小写）的测试
    # 其中可以包括使用文件名、类名和函数名作为变量的 Python 运算符
    pytest -k "MyClass and not method"
    # 运行文件中的单个测试用例
    pytest test_mod.py::test_func
    pytest test_mod.py::TestClass::test_method
    # 运行所有使用 ​@pytest.mark.slow​ 装饰器装饰的测试
    pytest -m slow
    # 在第一次失败时停止
    pytest -x
    # 在两次失败后停止
    pytest --maxfail=2
    # 输出：测试会话进度、测试失败时的断言详细信息、带有 ​--fixtures​ 的固定详细信息等
    pytest -v
    # 指定日志文件
    pytest --resultlog=path


请求fixture
----------------------
在测试中，​fixture​为测试 提供了一个定义好的、可靠的和一致的上下文。这可能包括环境（例如配置有已知参数的数据库）或内容（例如数据集）。
​Fixtures ​定义了构成测试排列阶段的步骤和数据。在 pytest 中，它们是您定义的用于此目的的函数

.. code-block:: python 

    import pytest

    class Fruit:
        def __init__(self, name):
            self.name = name
            self.cubed = False

        def cube(self):
            self.cubed = True


    class FruitSalad:
        def __init__(self, *fruit_bowl):
            self.fruit = fruit_bowl
            self._cube_fruit()

        def _cube_fruit(self):
            for fruit in self.fruit:
                fruit.cube()


    # Arrange
    @pytest.fixture
    def fruit_bowl():
        return [Fruit("apple"), Fruit("banana")]


    def test_fruit_salad(fruit_bowl):
        # Act
        fruit_salad = FruitSalad(*fruit_bowl)

        # Assert
        assert all(fruit.cubed for fruit in fruit_salad.fruit)

* Fixtures可以请求其他fixtures
* Fixtures可重复使用
* 一个test/fixture一次可以请求多个fixture
* 每个测试可以请求fixture多次(缓存返回值)


自动适配fixture
-----------------------------
有时，您可能希望拥有一个(甚至几个)​​fixture​​，您知道所有的测试都将依赖于它。​
​autouse fixture​​是使所有测试自动请求它们的一种方便的方法。这可以减少大量的冗余请求，甚至可以提供更高级的​​fixture​​使用。

我们可以将​​autouse =True​​传递给​​fixture​​的装饰器，从而使一个​​fixture​​成为​​autouse fixture​

.. code-block:: python

    import pytest

    @pytest.fixture
    def first_entry():
        return "a"

    @pytest.fixture
    def order(first_entry):
        return []

    @pytest.fixture(autouse=True)
    def append_first(order, first_entry):
        return order.append(first_entry)

    def test_string_only(order, first_entry):
        assert order == [first_entry]

    def test_string_and_int(order, first_entry):
        order.append(2)
        assert order == [first_entry, 2]

在本例中，​​append_first fixture​​是一个自动使用的​​fixture​​。因为它是自动发生的，所以两个测试都受到它的影响，即使没有一个测试请求它。
但这并不意味着他们不能提出要求，只是说没有必要


用属性标记测试函数
--------------------------------
通过使用 pytest.mark 助手，您可以轻松地在测试函数上设置元数据，或者，您可以使用CLI - pytest --markers列出所有标记，包括内置标记和自定义标记。

以下是一些内置标记：

* usefixtures——在测试函数或类上使用fixture
* filterwarnings—过滤测试函数的某些警告
* skip—总是跳过一个测试函数
* skipif-如果满足某个条件，则跳过某个测试函数
* Xfail——如果满足某个条件，则产生一个“预期失败”的结果
* parametrize——对同一个测试函数执行多个调用

.. code-block:: python 

    import pytest


    testcases = [
        ('111', 'aaa'),
        ('222', 'bbb'),
        ('333', 'ccc'),
        ('444', 'ddd'),
        ('555', 'eee'),
    ]

    @pytest.mark.parametrize('test_input, expected', testcases)
    def test_data_set(test_input, expected):
        print(test_input,expected)
        assert 1

