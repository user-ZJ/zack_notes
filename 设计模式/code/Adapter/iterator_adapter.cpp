//=========== back_insert_iterator

template <class _Container>
class  back_insert_iterator
    : public iterator<output_iterator_tag, void, void, void, void>
{
protected:
    _Container* container;
public:

    explicit back_insert_iterator(_Container& __x) : 
        container(std::addressof(__x)) {

    }
    back_insert_iterator& operator=(const typename _Container::value_type& __value_)
    {
        container->push_back(__value_); 
        return *this;
    }
    back_insert_iterator& operator=(typename _Container::value_type&& __value_)
    {
        container->push_back(std::move(__value_)); 
        return *this;
    }
    back_insert_iterator& operator*()     {return *this;}
    back_insert_iterator& operator++()    {return *this;}
    back_insert_iterator  operator++(int) {return *this;}

    typedef output_iterator_tag iterator_category;
    typedef void value_type;
    typedef ptrdiff_t difference_type;
    typedef void pointer;
    typedef void reference;
    typedef _Container container_type;
};

template <class _Container>
back_insert_iterator<_Container> back_inserter(_Container& __x)
{
    return back_insert_iterator<_Container>(__x);
}


//=========== front_insert_iterator


template <class _Container>
class  front_insert_iterator
    : public iterator<output_iterator_tag, void, void, void, void>
{
protected:
    _Container* container;
public:
    typedef output_iterator_tag iterator_category;
    typedef void value_type;
    typedef ptrdiff_t difference_type;
    typedef void pointer;
    typedef void reference;
    typedef _Container container_type;

    explicit front_insert_iterator(_Container& __x) : 
    container(std::addressof(__x)) {}
    front_insert_iterator& operator=(const typename _Container::value_type& __value_)
    {
        container->push_front(__value_); 
        return *this;
    }
    front_insert_iterator& operator=(typename _Container::value_type&& __value_)
    {
        container->push_front(std::move(__value_)); 
        return *this;
    }
    front_insert_iterator& operator*()     {return *this;}
    front_insert_iterator& operator++()    {return *this;}
    front_insert_iterator  operator++(int) {return *this;}
};



template <class _Container>
inline front_insert_iterator<_Container> front_inserter(_Container& __x)
{
    return front_insert_iterator<_Container>(__x);
}


//=========== insert_iterator


template <class _Container>
class  insert_iterator
    : public iterator<output_iterator_tag, void, void, void, void>
{
protected:
    _Container* container;
    typename _Container::iterator iter; 
public:
    typedef output_iterator_tag iterator_category;
    typedef void value_type;
    typedef ptrdiff_t difference_type;
    typedef void pointer;
    typedef void reference;
    typedef _Container container_type;

    insert_iterator(_Container& __x, typename _Container::iterator __i)
        : container(std::addressof(__x)), iter(__i) {}

    insert_iterator& operator=(const typename _Container::value_type& __value_)
    {
        iter = container->insert(iter, __value_); 
        ++iter; 
        return *this;
    }
     insert_iterator& operator=(typename _Container::value_type&& __value_)
    {
        iter = container->insert(iter, std::move(__value_)); 
        ++iter; 
        return *this;
    }

    insert_iterator& operator*()        {return *this;}
    insert_iterator& operator++()       {return *this;}
    insert_iterator& operator++(int)    {return *this;}
};

template <class _Container>
inline 
insert_iterator<_Container>
inserter(_Container& __x, typename _Container::iterator __i)
{
    return insert_iterator<_Container>(__x, __i);
}


工厂方法 Factory Method
抽象工厂 Abstract Factory
单件 Singleton
生成器 Builder
原型 Prototype

组合 Composite
装饰 Decorator
桥接 Bridge
适配器 Adapter
代理 Proxy
外观 Facade
享元 Flyweight

模板方法 Template Method
策略 Strategy
观察者 Observer
迭代器 Iterator
命令 Command
状态 State
职责链 Chain of Responsibility
解释器 Interpreter
中介者 Mediator
备忘录 Memento
访问器 Visitor

