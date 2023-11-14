flatbuffers使用笔记
=============================

编译
---------------
.. code-block:: shell

    git clone https://github.com/google/flatbuffers.git
    cd flatbuffers
    mkdir build
    cd build
    cmake ..
    make 

生成代码
------------------
.. code-block:: shell

    ./flatc --cpp monster.fbs


schema
----------------

.. code-block:: 

    namespace MyGame;
    attribute "priority";
    enum Color : byte { Red = 1, Green, Blue }
    union Any { Monster, Weapon, Pickup }

    struct Vec3 {
        x:float;
        y:float;
        z:float;
    }

    table Monster {
        pos:Vec3;
        mana:short = 150;
        hp:short = 100;
        name:string;
        friendly:bool = false (deprecated, priority: 1);
        inventory:[ubyte];
        color:Color = Blue;
        test:Any;
    }

    root_type Monster;

Tables
`````````````
Tables是在 FlatBuffers 中定义对象的主要方式，由名称和字段列表组成。
每个字段都有一个名称、一个类型和一个可选的默认值。如果未指定默认值，则默认值将用于0类型或null类型。

注意：

* 只能在表定义的末尾在架构中添加新字段。较旧的数据仍将正确读取，并在读取时为您提供默认值。旧代码将简单地忽略新字段。如果您希望灵活地在架构中使用任何顺序的字段，您可以手动分配 id（很像协议缓冲区）
* 无法从架构中删除不再使用的字段，但您可以简单地停止将它们写入数据中以获得几乎相同的效果。此外，您可以将它们标记为deprecated如上例所示，这将阻止在生成的 C++ 中生成访问器，作为强制不再使用该字段的方法。（小心：这可能会破坏代码！）。


Structs
```````````````````
和Tables类似，只是现在没有任何字段是可选的（因此也没有默认值），并且不能添加或弃用字段。结构体只能包含标量或其他结构体。将其用于简单对象，您非常确定不会进行任何更改（示例中非常清楚Vec3）。结构比表使用更少的内存，并且访问速度更快（它们总是内联存储在其父对象中，并且不使用虚拟表）

内置类型
`````````````````
* 8 bit: byte (int8), ubyte (uint8), bool
* 16 bit: short (int16), ushort (uint16)
* 32 bit: int (int32), uint (uint32), float (float32)
* 64 bit: long (int64), ulong (uint64), double (float64)

* 任何其他类型的向量（用 表示[type]）。不支持嵌套向量，您可以将内部向量包装在表中。
* string，它只能保存 UTF-8 或 7 位 ASCII。对于其他文本编码或一般二进制数据，请使用向量 ([byte]或[ubyte]) 代替。

数组
`````````````
.. code-block:: 

    struct Vec3 {
        v:[float:3];
    }

目前仅支持数组struct。


默认值、可选值和必需值
```````````````````````````````````
只有标量值可以有默认值，非标量（字符串/向量/表）字段在不存在时默认为null。

Enums
`````````````````
定义一系列命名常量，每个常量都有一个给定值，或者比前一个值增加 1。默认的第一个值为0。

可以使用（在本例中为）指定枚举的基础整型类型byte，然后该类型确定使用此枚举类型声明的任何字段的类型。仅允许使用整数类型，即byte、ubyte、short ushort、int、uint、long、ulong。

通常，枚举值只能被添加，不能被删除（枚举不会被弃用）


Unions
`````````````
可以声明一个联合字段，它可以保存对任何这些类型的引用，并且还会生成一个带有_type后缀的字段，该字段保存相应的枚举值，使您可以知道在运行时要转换为哪种类型。

联合包含一个特殊NONE标记来表示不存储任何值，因此名称不能用作别名。


属性
--------------
* id: n（在表字段上）：手动将字段标识符设置为n。如果使用此属性，则必须在此表的所有字段上使用它，并且数字必须是从 0 开始的连续范围。此外，由于联合类型有效地添加了两个字段，因此其 id 必须是第二个字段的 id（第一个字段是类型字段，未在架构中显式声明）。例如，如果联合字段之前的最后一个字段的 ID 为 6，则联合字段的 ID 应该为 8，并且联合类型字段将隐式为 7。ID 允许在架构中以任意顺序放置字段。当新字段添加到架构中时，它必须使用下一个可用的 ID。
* deprecated（在字段上）：不再为此字段生成访问器，代码应停止使用此数据。旧数据可能仍包含此字段，但新代码将无法再访问它。请注意，如果您弃用以前必需的字段，旧代码可能无法验证新数据（使用可选验证器时）。
* required（在非标量表字段上）：必须始终设置此字段。默认情况下，字段不需要出现在二进制文件中。这是可取的，因为它有助于向前/向后兼容性以及数据结构的灵活性。通过指定此属性，您可以使读取器和写入器都不出现错误。读取代码可以直接访问该字段，而不检查是否为空。如果构造代码没有初始化该字段，它们将得到一个断言，并且验证器将在缺少必需字段的缓冲区上失败。添加和删​​除此属性都可能向前/向后不兼容，因为读取器将无法分别读取旧数据或新数据，除非数据碰巧始终具有该字段集。
* force_align: size（在结构体上）：强制该结构的对齐方式高于其自然对齐的方式。导致这些结构与缓冲区内的该数量对齐，如果该缓冲区是按照该对齐方式分配的（对于直接在 a 内部访问的缓冲区来说不一定是这种情况FlatBufferBuilder）。注意：目前不保证与 一起使用时有效--object-api，因为这可能会在比您指定的对齐方式少的位置分配对象force_align。
* force_align: size（在向量上）：强制该向量的对齐方式与元素大小通常规定的对齐方式不同。注意：现在仅适用于生成的 C++ 代码。
* bit_flags（在无符号枚举上）：此字段的值表示位，这意味着模式中指定的任何无符号值 N 最终将表示 1<<N，或者如果您根本不指定值，您将得到序列 1, 2, 4, 8, ...
* nested_flatbuffer: "table_name"（在字段上）：这表明该字段（必须是 ubyte 的向量）包含 Flatbuffer 数据，其根类型由 给出table_name。然后，生成的代码将为嵌套的 FlatBuffer 生成一个方便的访问器。
* flexbuffer（在字段上）：这表明该字段（必须是 ubyte 的向量）包含 Flexbuffer 数据。然后，生成的代码将为 FlexBuffer 根生成一个方便的访问器。
* key（在字段上）：该字段在对其所在表类型的向量进行排序时用作键。可用于就地二分搜索。
* hash（在字段上）。这是一个（无）符号的 32/64 位整数字段，在 JSON 解析期间允许其值是一个字符串，然后将其存储为其哈希值。属性的值是要使用的哈希算法，其中之一fnv1_32 fnv1_64 fnv1a_32 fnv1a_64。
* original_order（在表上）：由于表中的元素不需要以任何特定顺序存储，因此通常通过按大小排序来优化空间。这个属性可以阻止这种情况发生。通常不应该有任何理由使用此标志。
* ``native_*`` 添加了几个属性来支持“基于C++ 对象的 API”。所有这些属性都以术语 ``native_`` 为前缀。


JSON解析
-------------------
sample.fbs 

.. code-block:: 

    table sample
    {
        firstName: string;
        lastName: string;
        age: int;
    }

    root_type sample;

test.cpp:

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include "flatbuffers/idl.h"

    int main()
    {
        std::string input_json_data = "{first_name: \"somename\",last_name: \"someothername\",age: 21}";

        std::string schemafile;
        std::string jsonfile;
        bool ok = flatbuffers::LoadFile("sample.fbs", false, &schemafile);
        if (!ok) {
            std::cout << "load file failed!" << std::endl;
            return -1;
        }
        std::cout<<"schemafile"<<schemafile<<std::endl;

        flatbuffers::Parser parser;
        parser.Parse(schemafile.c_str());
        if (!parser.Parse(input_json_data.c_str())) {
            std::cout << "flatbuffers parser failed with error : " << parser.error_ << std::endl;
            return -1;
        }

        std::string jsongen;
        if (GenText(parser, parser.builder_.GetBufferPointer(), &jsongen)) {
            std::cout << "Couldn't serialize parsed data to JSON!" << std::endl;
            return -1;
        }

        std::cout << "intput json" << std::endl
                << input_json_data << std::endl
                << std::endl
                << "output json" << std::endl
                << jsongen << std::endl;

        return 0;
    }

CMakeLists.txt

.. code-block:: cmake

    include_directories(
        flatbuffers/include
    )
    link_directories(
        flatbuffers/build/
    )

    add_executable(test test.cpp)
    target_link_libraries(test flatbuffers)


https://github.com/google/flatbuffers/blob/master/samples/sample_text.cpp


https://stackoverflow.com/questions/48215929/can-i-serialize-dserialize-flatbuffers-to-from-json
