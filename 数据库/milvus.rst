milvus
======================

milvus文档： https://milvus.io/docs/install_standalone-docker.md

https://www.bilibili.com/video/BV11a4y1c7SW/?spm_id_from=333.999.0.0&vd_source=1cbcdbb91c2e108ff4f290eeb865ee30

客户端：attu

安装
-------------------
.. code-block:: shell

    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
    bash standalone_embed.sh start



文本类型配置倒排索引
---------------------------------
https://milvus.io/docs/index-scalar-fields.md


示例
------------
.. code-block:: python 

    from pymilvus import __version__
    from pymilvus import MilvusClient
    from pymilvus import connections, db
    from pymilvus import FieldSchema,DataType,CollectionSchema
    from pymilvus import Collection
    import pandas as pd
    import random
    import time
    print(__version__)


    host = "10.12.51.190"
    port = 19530
    dbname = "book"

    conn = connections.connect(host=host, port=port)


    # 创建数据库
    if dbname not in db.list_database():
        database = db.create_database(dbname)
    else:
        db.using_database(dbname)
        # db.drop_database(dbname)
        # database = db.create_database(dbname)
        # db.using_database(dbname)

    conn = connections.connect(
        host=host,
        port=port,
        db_name=dbname
    )

    print(db.list_database())



    # 连接数据库
    client = MilvusClient(
        uri="http://{}:{}".format(host,port),
        db_name=dbname
    )

    # 查看数据库中collection
    print(client.list_collections())

    # 构造字段
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    # Add fields to schema
    schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name="my_text", datatype=DataType.VARCHAR, max_length=256)

    # 创建表
    client.create_collection(
        collection_name="customized_setup_2",
        schema=schema,
    )

    # 查看表的加载状态
    res = client.get_load_state(
        collection_name="customized_setup_2"
    )

    print(res)

    # 查看所有表
    print(client.list_collections())

    # 创建索引
    # Prepare an empty IndexParams object, without having to specify any index parameters
    index_params = client.prepare_index_params() 
    index_params.add_index(
        field_name="my_id",
        index_type="STL_SORT"
    )
    # 倒排索引
    index_params.add_index(
        field_name="my_text", # Name of the scalar field to be indexed
        index_type="INVERTED", # Type of index to be created. For auto indexing, leave it empty or omit this parameter.
        index_name="inverted_index" # Name of the index to be created
    )

    index_params.add_index(
        field_name="my_vector", 
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={ "nlist": 128 }
    )

    client.create_index(
        collection_name="customized_setup_2",
        index_params=index_params
    )

    res = client.describe_collection(
        collection_name="customized_setup_2"
    )

    print(res)

    # 查看所有索引
    res = client.list_collections()

    print(res)

    client.load_collection(
        collection_name="customized_setup_2"
    )

    res = client.get_load_state(
        collection_name="customized_setup_2"
    )

    print(res)

    # client.list_indexes(
    #     collection_name="test_scalar_index"  # Specify the collection name
    # )

数据库管理
---------------------------
.. code-block:: python

    from pymilvus import connections, db
    conn = connections.connect(host="127.0.0.1", port=19530)
    # 创建数据库
    database = db.create_database("my_database")
    # 切换数据库
    db.using_database("my_database")
    # 连接数据库
    conn = connections.connect(
        host="127.0.0.1",
        port="19530",
        db_name="my_database"
    )
    # 查看数据库列表
    db.list_database()
    # 删除数据库
    db.drop_database("my_database")

字段管理
--------------------------
.. code-block:: python 

    from pymilvus import FieldSchema
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
    age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector")
    # 使用分区键创建字段
    position_field = FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=256, is_partition_key=True)
    # 创建字段时指定默认值
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        # configure default value `25` for field `age`
        FieldSchema(name="age", dtype=DataType.INT64, default_value=25, description="age"),
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector")
    ]

    # 创建字段集合
    from pymilvus import FieldSchema, CollectionSchema
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
    age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector")
    # Enable partition key on a field if you need to implement multi-tenancy based on the partition-key field
    position_field = FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=256, is_partition_key=True)
    # Set enable_dynamic_field to True if you need to use dynamic fields. 
    schema = CollectionSchema(fields=[id_field, age_field, embedding_field], auto_id=False, enable_dynamic_field=True, description="desc of a collection")

    # 使用字段创建集合
    collection_name1 = "tutorial_1"
    collection1 = Collection(name=collection_name1, schema=schema, using='default', shards_num=2)

    # 使用数据字段创建集合中的字段
    import pandas as pd
    df = pd.DataFrame({
        "id": [i for i in range(nb)],
        "age": [random.randint(20, 40) for i in range(nb)],
        "embedding": [[random.random() for _ in range(dim)] for _ in range(nb)],
        "position": "test_pos"
    })

    collection, ins_res = Collection.construct_from_dataframe(
        'my_collection',
        df,
        primary_field='id',
        auto_id=False
        )

集合管理
-------------------------
.. code-block:: python 

    from pymilvus import MilvusClient, DataType
    client = MilvusClient(
        uri="http://localhost:19530"
    )
    #快速创建集合
    client.create_collection(
        collection_name="quick_setup",
        dimension=5
    )
    # 查看集合状态
    res = client.get_load_state(
        collection_name="quick_setup"
    )
    print(res)
    # Output
    # {
    #     "state": "<LoadState: Loaded>"
    # }
    
    #自定义创建集合
    # 构建字段
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    # 配置索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="my_id",
        index_type="STL_SORT"
    )
    index_params.add_index(
        field_name="my_vector", 
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={ "nlist": 128 }
    )
    # 创建并加载索引
    client.create_collection(
        collection_name="customized_setup_1",
        schema=schema,
        index_params=index_params
    )
    time.sleep(5)
    res = client.get_load_state(
        collection_name="customized_setup_1"
    )
    print(res)

    # 创建和构建索引分开
    client.create_collection(
        collection_name="customized_setup_2",
        schema=schema,
    )
    res = client.get_load_state(
        collection_name="customized_setup_2"
    )
    print(res)
    client.create_index(
        collection_name="customized_setup_2",
        index_params=index_params
    )
    res = client.get_load_state(
        collection_name="customized_setup_2"
    )
    print(res)

    # 查看集合
    res = client.describe_collection(
        collection_name="customized_setup_2"
    )
    print(res)

    # 查看数据库中所有集合
    res = client.list_collections()

    #加载集合
    client.load_collection(
        collection_name="customized_setup_2",
        replica_number=1 # Number of replicas to create on query nodes. Max value is 1 for Milvus Standalone, and no greater than `queryNode.replicas` for Milvus Cluster.
    )
    res = client.get_load_state(
        collection_name="customized_setup_2"
    )
    print(res)
    #释放集合
    client.release_collection(
        collection_name="customized_setup_2"
    )
    # 删除集合
    client.drop_collection(
        collection_name="quick_setup"
    )

查询
----------------------
.. code-block:: python 

    from pymilvus import MilvusClient, DataType
    client = MilvusClient(
        uri="http://localhost:19530"
    )
    # 查询集合中所有数据
    result = client.query(collection_name=collection_name,filter="id >= 0")


C++客户端
----------------------------
https://github.com/milvus-io/milvus-sdk-cpp

安装
`````````
.. code-block:: shell

    git clone https://github.com/milvus-io/milvus-sdk-cpp.git
    cd milvus-sdk-cpp
    bash scripts/install_deps.sh
    mkdir build
    cd build
    cmake ..
    make