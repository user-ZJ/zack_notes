mongodb-cxx使用
===================================

https://mongocxx.org/mongocxx-v3/tutorial/


示例
----------------
.. code-block:: cpp

    #include <iostream>
    #include <cassert>
    #include <vector>
    #include <bsoncxx/builder/stream/document.hpp>
    #include <mongocxx/client.hpp>
    #include <mongocxx/instance.hpp>
    #include <bsoncxx/json.hpp>

    using bsoncxx::builder::basic::kvp;
    using bsoncxx::builder::basic::make_array;
    using bsoncxx::builder::basic::make_document;

    int main() {
        // 1. 连接数据库
        mongocxx::instance inst{};  // This should be done only once.
        mongocxx::uri uri("mongodb://admin:admin@10.12.50.209:27017/admin");
        mongocxx::client client(uri);

        // 2. 访问数据库
        auto db = client["mydb"];
        // 3. 访问collection
        auto collection = db["test"];

        // 4. 创建document
        bsoncxx::document::value doc_value = make_document(
            kvp("name", "MongoDB"),
            kvp("type", "database"),
            kvp("count", 1),
            kvp("versions", make_array("v6.0", "v5.0", "v4.4", "v4.2", "v4.0", "v3.6")),
            kvp("info", make_document(kvp("x", 203), kvp("y", 102))));
        // 5. 访问document
        bsoncxx::document::view doc_view = doc_value.view();
        bsoncxx::document::element element = doc_view["name"];
        assert(element.type() == bsoncxx::type::k_string);
        auto name = element.get_string().value; // For C++ driver version < 3.7.0, use get_utf8()
        assert(0 == name.compare("MongoDB"));

        // 6. 插入document
        // 插入单个document
        // return core::v1::optional<mongocxx::v_noabi::result::insert_one>
        auto insert_one_result = collection.insert_one(make_document(kvp("i", 0)));
        assert(insert_one_result);  // Acknowledged writes return results.
        // 获取自动生成的_id
        auto doc_id = insert_one_result->inserted_id(); 
        assert(doc_id.type() == bsoncxx::type::k_oid);
        // 插入多个document
        std::vector<bsoncxx::document::value> documents;
        documents.push_back(make_document(kvp("i", 1)));
        documents.push_back(make_document(kvp("i", 2)));
        auto insert_many_result = collection.insert_many(documents);
        assert(insert_many_result);  // Acknowledged writes return results.
        auto doc0_id = insert_many_result->inserted_ids().at(0);
        auto doc1_id = insert_many_result->inserted_ids().at(1);
        assert(doc0_id.type() == bsoncxx::type::k_oid);
        assert(doc1_id.type() == bsoncxx::type::k_oid);

        // 7. 从Collection中查询数据
        // 查询单个document
        auto find_one_result = collection.find_one({});
        if (find_one_result) {
            // Do something with *find_one_result
        }
        assert(find_one_result);
        // 查找所用的document
        auto cursor_all = collection.find({});
        std::cout << "collection " << collection.name()
            << " contains these documents:" << std::endl;
        for (auto doc : cursor_all) {
            // Do something with doc
            assert(doc["_id"].type() == bsoncxx::type::k_oid);
            std::cout << bsoncxx::to_json(doc, bsoncxx::ExtendedJsonMode::k_relaxed) << std::endl;
        }
        std::cout << std::endl;
        // 条件查找
        auto find_one_filtered_result = collection.find_one(make_document(kvp("i", 0)));
        if (find_one_filtered_result) {
            // Do something with *find_one_filtered_result
        }
        auto cursor_filtered =
        collection.find(make_document(kvp("i", make_document(kvp("$gt", 0), kvp("$lte", 2)))));
        for (auto doc : cursor_filtered) {
            // Do something with doc
            assert(doc["_id"].type() == bsoncxx::type::k_oid);
        }

        // 更新document
        // 更新单个数据
        auto update_one_result =
            collection.update_one(make_document(kvp("i", 0)),
                                make_document(kvp("$set", make_document(kvp("foo", "bar")))));
        assert(update_one_result);  // Acknowledged writes return results.
        assert(update_one_result->modified_count() == 1);
        // 更新多个数据
        auto update_many_result =
            collection.update_many(make_document(kvp("i", make_document(kvp("$gt", 0)))),
                                    make_document(kvp("$set", make_document(kvp("foo", "buzz")))));
        assert(update_many_result);  // Acknowledged writes return results.
        assert(update_many_result->modified_count() == 2);

        // 删除document
        // 删除单个数据
        auto delete_one_result = collection.delete_one(make_document(kvp("i", 0)));
        assert(delete_one_result);  // Acknowledged writes return results.
        assert(delete_one_result->deleted_count() == 1);
        // 按条件删除数据
        auto delete_many_result =
        collection.delete_many({});
        assert(delete_many_result);  // Acknowledged writes return results.
        assert(delete_many_result->deleted_count() == 2);
        
        // 创建索引
        // 对于升序索引类型，指定为1。对于降序索引类型，指定为-1。
        auto index_specification = make_document(kvp("i", 1));
        collection.create_index(std::move(index_specification));

        // 删除colection
        collection.drop();
        return 0;
    }

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.2)
    project(mongodb_test)

    set(CMAKE_CXX_STANDARD 17)

    find_package(mongocxx REQUIRED)
    find_package(bsoncxx REQUIRED)

    include_directories(${LIBMONGOCXX_INCLUDE_DIR})
    include_directories(${LIBBSONCXX_INCLUDE_DIR})

    add_executable(mongodb_test main.cpp)

    target_link_libraries(mongodb_test PRIVATE mongo::mongocxx_shared)

