C++序列化库使用
==========================

https://github.com/thekvs/cpp-serializers


cereal
---------------------
https://github.com/USCiLab/cereal

https://github.com/LowCostCustoms/cereal-optional

.. literalinclude:: ../code/cereal_example.cpp
    :language: cpp


自定义map序列化后格式
`````````````````````````````````
https://uscilab.github.io/cereal/archive_specialization.html

.. code-block:: cpp

    namespace cereal {
    //! Saving for std::map<std::string, std::string> for text based archives
    // Note that this shows off some internal cereal traits such as EnableIf,
    // which will only allow this template to be instantiated if its predicates
    // are true
    template <
        class Archive, class T,class C, class A,
        traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
    inline void save(Archive &ar,
                    std::map<std::string, T, C, A> const &map) {
    for (const auto &i : map)
        ar(cereal::make_nvp(i.first, i.second));
    }

    //! Loading for std::map<std::string, std::string> for text based archives
    template <
        class Archive, class T,class C, class A,
        traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
    inline void load(Archive &ar, std::map<std::string, T, C, A> &map) {
    map.clear();

    auto hint = map.begin();
    while (true) {
        const auto namePtr = ar.getNodeName();

        if (!namePtr)
        break;

        std::string key = namePtr;
        T value;
        ar(value);
        hint = map.emplace_hint(hint, std::move(key), std::move(value));
    }
    }
    } // namespace cereal
