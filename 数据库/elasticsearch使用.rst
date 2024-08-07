elasticsearch使用
=================================

https://www.elastic.co/guide/en/elasticsearch/reference/current/rest-apis.html


存储和查询向量数据
`````````````````````````````
https://www.elastic.co/guide/en/elasticsearch/reference/8.3/dense-vector.html

https://www.elastic.co/guide/en/elasticsearch/reference/8.3/knn-search.html#tune-approximate-knn-for-speed-accuracy

https://www.elastic.co/guide/en/elasticsearch/reference/8.13/rrf.html

索引操作
----------------

查询索引是否存在
`````````````````````````
.. code-block:: shell

    curl --location --head 'localhost:9200/test'

head请求，返回200说明索引存在，否则返回404

索引创建
`````````````````
.. code-block:: shell

    curl --location --request PUT 'localhost:9200/test' \
        --header 'Content-Type: application/json' \
        --data '{
            "mappings": {
                "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": false
                }
                }
            }
            }'

获取索引信息
```````````````````````````
.. code-block:: shell

    curl --location --request GET 'localhost:9200/test'
    # 获取所有索引信息
    curl --location --request GET 'localhost:9200/_cat/indices?v'

删除索引
```````````````````
.. code-block:: shell

    curl --location --request DELETE 'localhost:9200/test'

文档操作
---------------------------

插入文档
```````````````````
.. code-block:: shell

    # 随机id
    curl --location --request POST 'localhost:9200/test/_doc' \
        --header 'Content-Type: application/json' \
        --data '{
            "key1":"value1",
            "key2":"value2"
        }'
    # 指定id,在url后面添加id. 
    curl --location --request POST 'localhost:9200/test/_doc/1001' \
        --header 'Content-Type: application/json' \
        --data '{
            "key1":"value1",
            "key2":"value2"
        }'
    # 批量插入
    curl --location --request POST 'localhost:9200/_bulk' \
        --header 'Content-Type: application/json' \
        --data '{ "index" : { "_index" : "test", "_id" : "1" } }
            { "field1" : "value1" }
            { "delete" : { "_index" : "test", "_id" : "2" } }
            { "create" : { "_index" : "test", "_id" : "3" } }
            { "field1" : "value3" }
            { "update" : {"_id" : "1", "_index" : "test"} }
            { "doc" : {"field2" : "value2"} }'


查询文档
`````````````
.. code-block:: shell

    # 通过id查询
    curl --location --request GET 'localhost:9200/test/_doc/1001' 
    # 查询index中所有数据/全量查询
    curl --location --request GET 'localhost:9200/test/_search'
    # 条件查询
    curl --location --request GET 'localhost:9200/test/_search?q=key1:value1' 
    curl --location --request POST 'localhost:9200/test/_search' \
        --header 'Content-Type: application/json' \
        --data '{
            "query":{
                "match":{
                    "key1":"value1"
                }
            }
        }'

    # 多条件查询 must是与，should是或
    curl --location --request POST 'localhost:9200/test/_search' \
        --header 'Content-Type: application/json' \
        --data '{
            "query":{
                "bool":{
                    "must":[
                        {
                            "match":{
                                "key1":"value1"
                            }
                        },
                        {
                            "match":{
                                "key2":"value2"
                            }
                        }
                    ]
                }
                
            }
        }'

    # 全量查询
    curl --location --request POST 'localhost:9200/test/_search' \
        --header 'Content-Type: application/json' \
        --data '{
            "query":{
                "match_all":{}
            }
        }'

    # 分页查询
    curl --location --request POST 'localhost:9200/test/_search' \
        --header 'Content-Type: application/json' \
        --data '{
            "query":{
                "match_all":{}
            },
            "from":0,  起始位置
            "size":10  查询的数据量
        }'

    # 查询后字段筛选
    curl --location --request POST 'localhost:9200/test/_search' \
        --header 'Content-Type: application/json' \
        --data '{
            "query":{
                "match_all":{}
            },
            "_source":["key1"]  只查询key1字段
            "_source": {
                "excludes": ["field1", "field2"]  排除某些字段
            }
        }'

    # 查询后根据指定字段排序
    curl --location --request POST 'localhost:9200/test/_search' \
        --header 'Content-Type: application/json' \
        --data '{
            "query":{
                "match_all":{}
            },
            "sort":{
                "key1":{
                    "order":"asc"
                }
            }
        }'

    # 范围查询
    curl --location --request POST 'localhost:9200/test/_search' \
        --header 'Content-Type: application/json' \
        --data '{
            "query":{
                "bool":{
                    "filter":{
                        "range":{
                            "key1":{
                                "gt":300 大于300
                            }
                        }
                    }
                }
                
            }
        }'
    
    # 完全匹配
    # match 标识分词匹配
    # match_phrase表示完全匹配




修改文档
`````````````````````````
.. code-block:: shell

    # 直接覆盖数据
    curl --location --request PUT 'localhost:9200/test/_doc/1001' \
        --header 'Content-Type: application/json' \
        --data '{
            "key1":"value1",
            "key2":"value2"
        }'

    # 修改部分数据
    curl --location --request POST 'localhost:9200/test/_update/1001' \
        --header 'Content-Type: application/json' \
        --data '{
            "key1":"123"
        }'

    # 新增一个字段，是某两个字段的拼接
    curl --location --request POST 'localhost:9200/test/_update_by_query' \
        --header 'Content-Type: application/json' \
        --data '{
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "ctx._source.new_field = ctx._source.field1 + '\n' + ctx._source.field2",
                "lang": "painless" // 使用Painless脚本语言
            }
        }'


删除文档
`````````````````````
.. code-block:: shell

    # 根据id删除
    curl --location --request DELETE 'localhost:9200/test/_doc/1001'
    # 全量删除
    curl --location --request POST 'localhost:9200/test/_delete_by_query' \
        --header 'Content-Type: application/json' \
        --data '{"query": {"match_all": {}}}'


从一个数据库同步数据到另一个数据库
------------------------------------------------
工具：elasticdump

https://github.com/elasticsearch-dump/elasticsearch-dump

.. code-block:: shell

    docker run --rm -ti elasticdump/elasticsearch-dump \
        --input=http://production.es.com:9200/my_index \
        --output=http://staging.es.com:9200/my_index \
        --type=data