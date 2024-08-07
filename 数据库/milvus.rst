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

    # 创建索引
    index_params = client.prepare_index_params() # Prepare an empty IndexParams object, without having to specify any index parameters
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