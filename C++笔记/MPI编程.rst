MPI编程
=============================

OpenMPI安装
-------------------------
.. code-block:: shell

    wget -O /tmp/openmpi-3.0.4.tar.gz \
       https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.4.tar.gz
    tar xzf /tmp/openmpi-3.0.4.tar.gz -C /tmp
    cd /tmp/openmpi-3.0.4
    ./configure --enable-orterun-prefix-by-default --with-cuda=/usr/local/cuda
    make -j $(nproc) all && sudo make install
    sudo ldconfig
    mpirun --version


hello world
----------------------
.. code-block:: cpp

    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv) {
    // Initialize the MPI environment. The two arguments to MPI Init are not
    // currently used by MPI implementations, but are there in case future
    // implementations might need the arguments.
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, world_rank, world_size);

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
    }

.. code-block:: shell

    mpicc -o mpi_hello_world mpi_hello_world.c
    # 4个线程运行
    mpirun -n 4 ./mpi_hello_world


Send/Receive
----------------------------
.. code-block:: cpp

    MPI_Send(
        void* data,
        int count,
        MPI_Datatype datatype,
        int destination,
        int tag,
        MPI_Comm communicator);

    MPI_Recv(
        void* data,
        int count, //最多接收count个元素
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm communicator,
        MPI_Status* status);

.. note:: 

    使用--with-cuda编译，可以直接发送GPU内存地址


MPI_Datatype
```````````````````````
+------------------------+------------------------+
|      MPI datatype      |      C equivalent      |
+========================+========================+
| MPI_SHORT              | short int              |
+------------------------+------------------------+
| MPI_INT                | int                    |
+------------------------+------------------------+
| MPI_LONG               | long int               |
+------------------------+------------------------+
| MPI_LONG_LONG          | long long int          |
+------------------------+------------------------+
| MPI_UNSIGNED_CHAR      | unsigned char          |
+------------------------+------------------------+
| MPI_UNSIGNED_SHORT     | unsigned short int     |
+------------------------+------------------------+
| MPI_UNSIGNED           | unsigned int           |
+------------------------+------------------------+
| MPI_UNSIGNED_LONG      | unsigned long int      |
+------------------------+------------------------+
| MPI_UNSIGNED_LONG_LONG | unsigned long long int |
+------------------------+------------------------+
| MPI_FLOAT              | float                  |
+------------------------+------------------------+
| MPI_DOUBLE             | double                 |
+------------------------+------------------------+
| MPI_LONG_DOUBLE        | long double            |
+------------------------+------------------------+
| MPI_BYTE               | char                   |
+------------------------+------------------------+


MPI_Status
```````````````````````
* status.MPI_SOURCE:发送端进程的秩
* MPI_TAG:消息的标签
* MPI_Get_count(&status, MPI_INT, &number_amount)：消息的长度.
  不能保证 MPI_Recv 能够接收函数调用参数的全部元素。 相反，它只接收已发送给它的元素数量（如果发送的元素多于所需的接收数量，则返回错误。）

.. code-block:: cpp

    int number_amount;
    MPI_Status status;
    // Receive at most MAX_NUMBERS from process zero
    MPI_Recv(numbers, MAX_NUMBERS, MPI_INT, 0, 0, MPI_COMM_WORLD,&status);
    MPI_Get_count(&status, MPI_INT, &number_amount);
    printf("1 received %d numbers from 0. Message source = %d, tag = %d\n",
           number_amount, status.MPI_SOURCE, status.MPI_TAG);


MPI_Probe
-------------------------
除了传递接收消息并简易地配备一个很大的缓冲区来为所有可能的大小的消息提供处理，
还可以使用 MPI_Probe 在实际接收消息之前查询消息大小。 函数原型看起来像这样：

.. code-block:: cpp

    MPI_Probe(
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Status* status)

MPI_Probe 看起来与 MPI_Recv 非常相似。 实际上，您可以将 MPI_Probe 视为 MPI_Recv，除了不接收消息外，它们执行相同的功能。 
与 MPI_Recv 类似，MPI_Probe 将阻塞具有匹配标签和发送端的消息。 当消息可用时，它将填充 status 结构体。 然后，用户可以使用 MPI_Recv 接收实际的消息。

.. code-block:: cpp

    int number_amount;
    if (world_rank == 0) {
        const int MAX_NUMBERS = 100;
        int numbers[MAX_NUMBERS];
        // Pick a random amount of integers to send to process one
        srand(time(NULL));
        number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;

        // Send the random amount of integers to process one
        MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("0 sent %d numbers to 1\n", number_amount);
    } else if (world_rank == 1) {
        MPI_Status status;
        // Probe for an incoming message from process zero
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

        // When probe returns, the status object has the size and other
        // attributes of the incoming message. Get the message size
        MPI_Get_count(&status, MPI_INT, &number_amount);

        // Allocate a buffer to hold the incoming numbers
        int* number_buf = (int*)malloc(sizeof(int) * number_amount);

        // Now receive the message with the allocated buffer
        MPI_Recv(number_buf, number_amount, MPI_INT, 0, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("1 dynamically received %d numbers from 0.\n",
            number_amount);
        free(number_buf);
    }

MPI_Barrier
------------------------
同步点:所有的进程在执行代码的时候必须首先都到达一个同步点才能继续执行后面的代码

.. code-block:: cpp

    MPI_Barrier(MPI_Comm communicator)

Barrier，屏障- 这个方法会构建一个屏障，任何进程都没法跨越屏障，直到所有的进程都到达屏障。


MPI_Bcast
------------------------
广播 (broadcast) 是标准的集体通信技术之一。一个广播发生的时候，一个进程会把同样一份数据传递给一个 communicator 里的所有其他进程。
广播的主要用途之一是把用户输入传递给一个分布式程序，或者把一些配置参数传递给所有的进程。

.. image:: /images/C++/broadcast_pattern.png

.. code-block:: cpp

    MPI_Bcast(
        void* data,
        int count,
        MPI_Datatype datatype,
        int root,
        MPI_Comm communicator)

尽管根节点和接收节点做不同的事情，它们都是调用同样的这个 MPI_Bcast 函数来实现广播。
当根节点(在我们的例子是节点0)调用 MPI_Bcast 函数的时候，data 变量里的值会被发送到其他的节点上。
当其他的节点调用 MPI_Bcast 的时候，data 变量会被赋值成从根节点接受到的数据。

MPI_Scatter
------------------------
MPI_Scatter 是一个跟 MPI_Bcast 类似的集体通信机制。MPI_Scatter 的操作会设计一个指定的根进程，根进程会将数据发送到 communicator 里面的所有进程。
MPI_Bcast 和 MPI_Scatter 的主要区别很小但是很重要。MPI_Bcast 给每个进程发送的是同样的数据，然而 MPI_Scatter 给每个进程发送的是一个数组的一部分数据。

.. image:: /images/C++/broadcastvsscatter.png

.. code-block:: cpp

    MPI_Scatter(
        void* send_data,//根进程上的一个数据数组
        int send_count, // 发送给每个进程的数据数量
        MPI_Datatype send_datatype,//发送给每个进程的数据类型
        void* recv_data,
        int recv_count,
        MPI_Datatype recv_datatype,
        int root, //分发数组的根进程
        MPI_Comm communicator)

MPI_Gather
--------------------
MPI_Gather 跟 MPI_Scatter 是相反的。MPI_Gather 从好多进程里面收集数据到一个进程上面而不是从一个进程分发数据到多个进程。
这个机制对很多平行算法很有用，比如并行的排序和搜索。

.. image:: /images/C++/gather.png

跟MPI_Scatter类似，MPI_Gather从其他进程收集元素到根进程上面。元素是根据接收到的进程的秩排序的。

.. code-block:: cpp

    MPI_Gather(
        void* send_data,
        int send_count,
        MPI_Datatype send_datatype,
        void* recv_data,
        int recv_count,
        MPI_Datatype recv_datatype,
        int root,
        MPI_Comm communicator)

在MPI_Gather中，只有根进程需要一个有效的接收缓存。所有其他的调用进程可以传递NULL给recv_data。
另外，别忘记recv_count参数是从每个进程接收到的数据数量，而不是所有进程的数据总量之和。


MPI_Allgather
----------------------------
对于分发在所有进程上的一组数据来说，MPI_Allgather会收集所有数据到所有进程上。从最基础的角度来看，MPI_Allgather相当于一个MPI_Gather操作之后跟着一个MPI_Bcast操作

.. image:: /images/C++/allgather.png

就跟MPI_Gather一样，每个进程上的元素是根据他们的秩为顺序被收集起来的，只不过这次是收集到了所有进程上面。

MPI_Allgather的方法定义跟MPI_Gather几乎一样，只不过MPI_Allgather不需要root这个参数来指定根节点。

.. code-block:: cpp

    MPI_Allgather(
        void* send_data,
        int send_count,
        MPI_Datatype send_datatype,
        void* recv_data,
        int recv_count,
        MPI_Datatype recv_datatype,
        MPI_Comm communicator)

MPI_Reduce
----------------------
与 MPI_Gather 类似，MPI_Reduce 在每个进程上获取一个输入元素数组，并将输出元素数组返回给根进程。 输出元素包含减少的结果。

.. code-block:: cpp

    MPI_Reduce(
        void* send_data, 
        void* recv_data, 
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        int root,
        MPI_Comm communicator)

* send_data,  每个进程都希望归约的 datatype 类型元素的数组
* recv_data,  仅与具有 root 秩的进程相关，包含归约的结果，大小为sizeof（datatype）* count
* op 参数是您希望应用于数据的操作

+------------+------------------------------+
|     op     |             说明             |
+============+==============================+
| MPI_MAX    | 返回最大元素。               |
+------------+------------------------------+
| MPI_MIN    | 返回最小元素。               |
+------------+------------------------------+
| MPI_SUM    | 对元素求和。                 |
+------------+------------------------------+
| MPI_PROD   | 将所有元素相乘。             |
+------------+------------------------------+
| MPI_LAND   | 对元素执行逻辑与运算。       |
+------------+------------------------------+
| MPI_LOR    | 对元素执行逻辑或运算。       |
+------------+------------------------------+
| MPI_BAND   | 对元素的各个位按位与执行。   |
+------------+------------------------------+
| MPI_BOR    | 对元素的位执行按位或运算。   |
+------------+------------------------------+
| MPI_MAXLOC | 返回最大值和所在的进程的秩。 |
+------------+------------------------------+
| MPI_MINLOC | 返回最小值和所在的进程的秩。 |
+------------+------------------------------+

.. image:: /images/C++/mpi_reduce_1.png
.. image:: /images/C++/mpi_reduce_2.png

每个进程都有两个元素。 结果求和基于每个元素进行。 换句话说，不是将所有数组中的所有元素累加到一个元素中，而是将每个数组中的第 i 个元素累加到进程 0 结果数组中的第 i 个元素中。


MPI_Allreduce
--------------------
MPI_Allreduce 将归约值并将结果分配给所有进程。

.. code-block:: cpp

    MPI_Allreduce(
        void* send_data,
        void* recv_data,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm communicator)

MPI_Allreduce 与 MPI_Reduce 相同，不同之处在于它不需要根进程 ID（因为结果分配给所有进程）。

.. image:: /images/C++/mpi_allreduce_1.png


MPI_Comm_split
------------------------
MPI 允许您立即与通讯器中的所有进程进行对话，以执行诸如使用 MPI_Scatter 将数据从一个进程分发到多个进程或使用 MPI_Reduce 执行数据归约的操作。 
但是，到目前为止，我们仅使用了默认的通讯器 MPI_COMM_WORLD。

对于简单的应用程序，使用 MPI_COMM_WORLD 进行所有操作并不罕见，但是对于更复杂的用例，拥有更多的通讯器可能会有所帮助。 
例如，如果您想对网格中进程的子集执行计算。 例如，每一行中的所有进程都可能希望对一个值求和。 这将是第一个也是最常见的用于创建新的通讯器的函数：

.. code-block:: cpp

    MPI_Comm_split(
        MPI_Comm comm,
        int color,
        int key,
        MPI_Comm* newcomm)

MPI_Comm_split 通过基于输入值 color 和 key 将通讯器“拆分”为一组子通讯器来创建新的通讯器。 
在这里需要注意的是，原始的通讯器并没有消失，但是在每个进程中都会创建一个新的通讯器。 

* 第一个参数 comm 是通讯器，它将用作新通讯器的基础。 这可能是 MPI_COMM_WORLD，但也可能是其他任何通讯器。 
* 第二个参数 color 确定每个进程将属于哪个新的通讯器。 为 color 传递相同值的所有进程都分配给同一通讯器。 如果 color 为 MPI_UNDEFINED，则该进程将不包含在任何新的通讯器中。 
* 第三个参数 key 确定每个新通讯器中的顺序（秩）。 传递 key 最小值的进程将为 0，下一个最小值将为 1，依此类推。 如果存在平局，则在原始通讯器中秩较低的进程将是第一位。 
* 最后一个参数 newcomm 是 MPI 如何将新的通讯器返回给用户。


MPI_Comm_split_type
-------------------------------------
.. code-block:: cpp

    int MPI_Comm_split_type(MPI_Comm comm, int split_type, 
        int key, MPI_Info info, MPI_Comm *newcomm)

* comm​：表示输入的通信域（通常为一个已经创建好的communicator）。
* split_type​：表示划分类型，指定了如何划分communicator。目前MPI标准规定有以下划分类型：
  
  - MPI_COMM_TYPE_SHARED​：表示按照共享内存的方式划分。在具有共享内存的系统上，将进程划分到同一个共享内存区域中的communicator。
  - 其他自定义的划分类型：用户可以自己定义其他的划分类型，通过使用自定义的splitting function来实现特定的划分逻辑。
 
* key​：表示用于划分的键值。该键值与划分类型和通信域中每个进程的键值进行比较，决定进程在新的communicator中的划分。
* info​：表示可选的附加信息，通常用于进一步指定划分行为。
* newcomm​：表示输出参数，返回划分后的新communicator。

注意：MPI_Comm_split_type函数是一个collective操作，它需要在所有进程中同时调用，并且所有进程在调用结束后都能够得到相同的划分结果。



MPI_Comm_dup
------------------------
创建了一个通讯器的副本。





参考
------------------
https://mpitutorial.com/tutorials/


