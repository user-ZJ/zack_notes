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

