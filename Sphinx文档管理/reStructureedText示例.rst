====================
reStructuredText示例
====================

官方文档：https://www.sphinx-doc.org/zh_CN/master/contents.html

reStructuredText简介
=========================
Sphinx 是一个 **文档生成器** ，您也可以把它看成一种工具，它可以将一组纯文本源文件转换成各种输出格式，
并且自动生成交叉引用、索引等。也就是说，
如果您的目录包含一堆 `reStructuredText文件 <https://www.sphinx-doc.org/zh_CN/master/usage/restructuredtext/index.html>`_  
或 `Markdown <https://www.sphinx-doc.org/zh_CN/master/usage/markdown.html>`_  文档，
那么 Sphinx 就能生成一系列HTML文件，PDF文件（通过LaTeX），手册页等。


标题
=========
reStructuredText 中的“标题”被称为“Sections”，一般在文字下方加特殊字符以示区别：

特殊字符的重复长度应该大于等于标题（Sections）的长度。需要说明的是： reStructuredText 并不像 Markdown 那样，限定某一字符只表示特定的标题层级（比如 = 固定表示 H1 )。
而是解析器将遇到的第一个特殊字符渲染为 H1 ，第二个其它特殊字符渲染为 H2 ……以此类推。

::

    Section Title H1
    ================

    Section Title H2
    ----------------

    Section Title H3
    ````````````````

推荐使用的字符： ``= - ` ~ ' " ^ _ * + # : .``

当然，在 reStructuredText 的日常使用中，仍然建议养成习惯使用固定的特殊符号，方便别人一看到 = 就知道这是一级标题。 
除了 “Sections”外， reStructuredText 还支持“Title”和“SubTitle”，它们可以被配置为不在文档中出现。其实际作用更类似于“书名”，如《钢铁是怎样炼成的——保尔柯察金自传》。语法如下：

::

    ==================
    钢铁是怎样炼成的
    ==================

    ----------------
    保尔柯察金自传
    ----------------


区块引用
======================
区块引用使用空格或制表符的方式，一般是 4 个空格。   

    区块引用
    
        嵌套引用

段落内换行
============================
段落是构成reST文档的基本单位。通过一个或一个以上的空行隔开的文本区块就是一个段落。

注意在段落内换行并不会在html中生成换行符，要想保持在文本编辑器中的换行符，需要在这些行前面加上|和空格:

::

    | aaaaaaaa
    | bbbbbbbbb
    | cccccccccccc


| aaaaaaaa
| bbbbbbbbb
| cccccccccccc

.. note:: 

    如果编写中文reST文档，在编辑器中由于一行文字太长需要强制换行时，记得在行末加\\，
    否则生成的html会在行末和下一行行首之前插入一个空格

列表
==================
reStructuredText 支持有序列表和无序列表，语法与 Markdown 基本一致 

无序列表使用 ``- 、 * 、 +`` 来表示  

有序列表可以使用： 

1. 阿拉伯数字: 1, 2, 3, … (无上限)
2. 大写字母: A-Z
3. 小写字母: a-z
4. 大写罗马数字: I, II, III, IV, …, MMMMCMXCIX (4999)
5. 小写罗马数字: i, ii, iii, iv, …, mmmmcmxcix (4999)

todo
=================
在conf.py中添加

.. code-block:: python

    extensions = ['sphinx.ext.todo']
    todo_include_todos = True

* todo:待办项
* todolist:文档中所有todo的集合

::

    .. todolist::

    .. todo::

      * 待办项1

    .. todo::

      * 待办项2



代码块
=============
在reST文档中列出代码有三种方式：

1. 行内代码 用``code``
2. 简单代码块 在代码块的上一个段落后面加2个冒号，空一行后开始代码块，代码块要缩进
3. 复杂代码块 使用code-block指导语句，还可以选择列出行号和高亮重点行等
    * :linenos:显示行号
    * :emphasize-lines:3,6 3,6行高亮

行内代码
---------------
:: 

    ``echo "Hello World!";``

``echo "Hello World!";``

双冒号方式
-----------------
::

    ::
        
        echo "Hello World!";

::

        echo "Hello World!";

code-block 方式
-------------------------
::

    .. code-block:: python
        :emphasize-lines: 3,5
        :linenos:

        def some_function():
            interesting = False
            print 'This line is highlighted.'
            print 'This one is not...'
            print '...but this one is.'

.. code-block:: python
    :emphasize-lines: 3,5
    :linenos:

    def some_function():
        interesting = False
        print 'This line is highlighted.'
        print 'This one is not...'
        print '...but this one is.'

literalinclude方式
----------------------
::

    .. literalinclude:: 文件路径
        :language: cpp

    linenos：指定是否显示行号，默认为不显示。设置为 linenos=True 可以显示行号。
    start-after 和 end-before：指定在哪些文本行之后开始导入和在哪些文本行之前结束导入。
            这些参数允许你从源代码文件中选择性地导入部分代码。
            例如，设置start-after=//BEGIN CODE和end-before=//END CODE可以只导入这两个注释之间的代码。
    dedent：指定是否要自动减少导入的代码块的缩进。默认为 dedent=True。
    encoding：指定源文件的编码方式。如果不指定，则使用默认编码方式。

.. literalinclude:: test.cpp
   :language: cpp


数学公式
===========
支持latex数学公式表示，分为行内公式和单独行公式
  
行内公式表示为：``空格+：math:`公式`空格``

圆的面积为 :math:`A_\text{c} = (\pi/4) d^2` .

单行公式表示为：``.. math:: 换行+两个空格+公式``

.. code:: text

  .. math:: 
    \alpha _t(i) = P(O_1, O_2, \ldots  O_t, q_t = S_i \lambda )

.. math::
  \alpha _t(i) = P(O_1, O_2, \ldots  O_t, q_t = S_i \lambda )


分割线
===================
与 Markdown 语法基本一致：
::

-----------------------------------------

效果如下：

---------------------------------------------------------


链接
==============

参考式链接
------------------------
::

    欢迎访问 reStructuredText_ 官方主页。

    .. _reStructuredText: http://docutils.sf.net/

    如果是多个词组或者中文链接文本，则使用 ` 将其括住，就像这样：

    欢迎访问 `reStructuredText 结构化文本`_ 官方主页。

    .. _`reStructuredText 结构化文本`: http://docutils.sf.net/

欢迎访问 reStructuredText_ 官方主页。

.. _reStructuredText: http://docutils.sf.net/

欢迎访问 `reStructuredText 结构化文本`_ 官方主页。

.. _`reStructuredText 结构化文本`: http://docutils.sf.net/

行内式链接
-------------------------------
::

    `Python 编程语言 <http://www.python.org/>`_ 其实也有一些缺陷。

`Python 编程语言 <http://www.python.org/>`_ 其实也有一些缺陷。

自动标题链接
------------------------------------
reStructuredText 文档的各级标题（Sections）会自动生成链接，就像 GFM 风格的 Markdown 标记语言一样。
这在 reStructuredText 语法手册中被称为“隐式链接（Implicit Hyperlink）”。无论名称为何，我们将可以在文档中快速跳转到其它小节（Sections）

::

    本小节内容应该与 `行内标记`_ 结合学习。

本小节内容应该与 `行内标记`_ 结合学习。

rst文档链接
--------------------
::

    使用路径引用rst文档
    自定义引用文字
    :doc:`自定义名称为readthedocs <./readthedocs托管文档>`
    使用标题文字
    :doc:`./readthedocs托管文档`

:doc:`自定义名称为readthedocs <./readthedocs托管文档>`

:doc:`./readthedocs托管文档`

::

    使用标签引用文档
    :ref:`自定义名称为readthedocs <readthedocs托管文档>`
    :ref:`readthedocs托管文档`

    注意，需要再被引用的文件中添加 ".. _readthedocs托管文档:",否则不能被引用

:ref:`自定义名称为readthedocs <readthedocs托管文档>`

:ref:`readthedocs托管文档`

非rst文档链接
--------------------
会呈现出点击后下载文件的效果。注意这种引用方式在生成pdf文件时链接会无效。
::

    :download:`引用非rst的本地文档 <./download.zip>`

:download:`引用非rst的本地文档 <./download.zip>`

强调
====================
与 Markdown 语法基本相同。参看 `行内标记`_  

图片
=====================
reStructuredText 使用指令（Directives)的方式来插入图片。指令（Directives）作为 reStructuredText 语言的一种扩展机制，允许快速添加新的文档结构而无需对底层语法进行更改。

::

    .. image:: /images/nikola.png
        :align: center
        :width: 236px
        :height: 100px

.. image:: /images/nikola.png
   :align: center
   :width: 236px
   :height: 100px

插入图片的另一种方法是使用 figure 指令。该指令与 image 基本一样，不过可以为图片添加标题和说明文字。
两个指令共有的一个选项为 target ，可以为图片添加可点击的链接，甚至链接到另一张图片。那么结合 Nikola 博客的特定主题，就可以实现点击缩略图查看原图的效果

::

    .. figure:: /images/icarus.thumbnail.jpg
        :align: center
        :target: /images/icarus.jpg

        *飞向太阳*

.. figure:: /images/icarus.thumbnail.jpg
   :align: center
   :target: /images/icarus.jpg

   *飞向太阳*

表格
====================
::

    +------------------------+------------+----------+----------+
    | Header row, column 1   | Header 2   | Header 3 | Header 4 |
    | (header rows optional) |            |          |          |
    +========================+============+==========+==========+
    | body row 1, column 1   | column 2   | column 3 | column 4 |
    +------------------------+------------+----------+----------+
    | body row 2             | Cells may span columns.          |
    +------------------------+------------+---------------------+
    | body row 3             | Cells may  | - Table cells       |
    +------------------------+ span rows. | - contain           |
    | body row 4             |            | - body elements.    |
    +------------------------+------------+---------------------+

显示效果为：

+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | Cells may span columns.          |
+------------------------+------------+---------------------+
| body row 3             | Cells may  | - Table cells       |
+------------------------+ span rows. | - contain           |
| body row 4             |            | - body elements.    |
+------------------------+------------+---------------------+


这种表格语法被称为 Grid Tables 。如上所见， 
`Grid Tables <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#grid-tables>`_ 支持跨行跨列。

**在Grid Tables中支持"|"**

方法1：通过在之前添加一个额外的空格来移动文本;

方法2：添加一个额外的行

*注意：表格中有竖线，则不能使用vscode的Table Formatter进行格式化*

::
    
    +--------------+----------+-----------+-----------+
    | row 1, col 1 | column 2 | column 3  | column 4  |
    +==============+==========+===========+===========+
    | row 2        |  Use the command ``ls | more``.  |
    +--------------+----------+-----------+-----------+
    | row 3        |  ``|``   |           |           |
    |              |          |           |           |
    +--------------+----------+-----------+-----------+


如果你使用的编辑器创建该表格有困难，reStructuredText 还提供 Simple Tables 表格语法：

::

    =====  =====  ======
    Inputs     Output
    ------------  ------
    A      B    A or B
    =====  =====  ======
    False  False  False
    True   True   True
    =====  =====  ======

显示效果为：

=====  =====  ======
   Inputs     Output
------------  ------
  A      B    A or B
=====  =====  ======
False  False  False
True   True   True
=====  =====  ======

行内标记
===================

+------------------+--------------+----------------------------------------------------+
|       文本       |     结果     |                        说明                        |
+==================+==============+====================================================+
| \*强调\*         | *强调*       | 一般被渲染为斜体                                   |
+------------------+--------------+----------------------------------------------------+
| \*\*着重强调\*\* | **着重强调** | 一般被渲染为加粗                                   |
+------------------+--------------+----------------------------------------------------+
| \`解释文本\`     | `解释文本`   | 一般用于专用名词、文本引用、说明性文字等           |
+------------------+--------------+----------------------------------------------------+
| \`\`原样文本\`\` | ``原样文本`` | 与上面的区别在于：不会被转义。可用于行内代码书写。 |
+------------------+--------------+----------------------------------------------------+


目录
==================
https://zh-sphinx-doc.readthedocs.io/en/latest/markup/toctree.html

toctree
-------------------------
toctree 的用法，可以参考如下 `ReStructuredText 快速教程 <https://rst-tutorial.readthedocs.io/zh/latest/index.html#topics-index>`_ 的文档源码即可。

toctree指令会在当前位置插入文档的目录树。关联文档的路径可以使用相对路径或者绝对路径。
相对路径是指相对于toctree指令所在文件的路径。绝对路径是相对于源文件目录的路径。

toctree参数:

.. code:: text

    :maxdepth:2             指明了目录的层数，默认是包含所有的层。
    :numbered:              自动给章节添加编号
    :caption:               指定目录树的标题
    :name:                  名字，以便使用ref引用
    :titlesonly:            只显示文档的一级标题
    :glob:                  设置glob后，可以使用unix通配符匹配文档
    :reversed:              反向编号
    :hidden:                如果你只想使用最顶层的toctree，而忽略掉其它的toctree指令


文本颜色
============================
思路:

1. 首先在Rst中有一个 .. raw:: 标记, 可以引用纯Html. 所以我们可以将CSS语法放到该标记下.
2. 然后在Rst中有一个 .. role:: 标记, 可以为文档元素指定Html标签属性, 比如class属性.

如果我们定义了CSS, 又将CSS中的风格用Class属性关联起来, 那么我们就可以用自定义的RST Directives指定任何我们自定义的风格了.

.. role:: red
    :class: red

.. role:: blue
    :class: blue

.. role:: green
    :class: green

.. raw:: html

    <style>

    .red {
        color:red;
    }
    .blue {
        color:blue;
    }
    .green {
        color:green;
    }

    </style>


- This is :red:`Red` text.
- This is :blue:`Blue` text.
- This is :green:`Green` text.



指示性信息
=======================

.. attention:: 
    
    this is a attention message

.. caution:: 

    this is a caution message

.. danger:: 

    this is a danger message

.. error:: 

    this is a error message 

.. hint:: 

    this is a hint message 

.. important:: 

    this is a important message 

.. note:: 

    this is a note message 

.. tip:: 

    this is a tip message 

.. warning:: 

    | this is a warning message
    | this is a warning message



参考
=================
https://docutils.sourceforge.io/docs/user/rst/quickref.html

https://macplay.github.io/posts/cong-markdown-dao-restructuredtext/#id10

https://3vshej.cn/rstSyntax/index.html

https://hzz-rst.readthedocs.io/zh_CN/latest/index.html

https://cloud.tencent.com/developer/article/1195732

https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#code-examples


https://iridescent.ink/HowToMakeDocs/Basic/index.html
