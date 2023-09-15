#include <iostream>
#include <iterator>
#include <vector>
#include <typeinfo>

using namespace std;

/*
随机访问迭代器 random_access_iterator
（*） 、（->）、 （[n]）++、--、+、-、+=n、-=n、==、!=、<、>、<=、>=

双向迭代器bidirectional_iterator
（*） 、 （->）、++、--、==、!=

单向迭代器forward_iterator
（*） 、 （->）、++、==、!=

输入迭代器input_iterator
（*） 、（->）、++、==、!=

输出迭代器output_iterator
（*）++
*/


template<typename _Iterator, typename _Container>
class __normal_iterator
{
protected:
    _Iterator _M_current;
    typedef std::iterator_traits<_Iterator>		__traits_type;
public:
    typedef _Iterator  iterator_type;
    typedef typename __traits_type::iterator_category iterator_category;
    typedef typename __traits_type::value_type  	value_type;
    typedef typename __traits_type::difference_type 	difference_type;
    typedef typename __traits_type::reference 	reference;
    typedef typename __traits_type::pointer   	pointer;


     __normal_iterator() : _M_current(_Iterator()) { }

    explicit  __normal_iterator(const _Iterator& __i) : _M_current(__i) { }

    reference operator*() const { return *_M_current; }
    pointer operator->() const { return _M_current; }

    __normal_iterator& operator++() 
    {
        ++_M_current;
        return *this;
    }

    __normal_iterator operator++(int) 
    { return __normal_iterator(_M_current++); }

    __normal_iterator& operator--() 
    {
        --_M_current;
        return *this;
    }

    __normal_iterator operator--(int) 
    { return __normal_iterator(_M_current--); }

    reference operator[](difference_type __n) const 
    { return _M_current[__n]; }

    __normal_iterator& operator+=(difference_type __n) 
    { 
        _M_current += __n; 
        return *this; 
    }

    __normal_iterator operator+(difference_type __n) const 
    { 
        return __normal_iterator(_M_current + __n); 
    }

    __normal_iterator& operator-=(difference_type __n) 
    { 
        _M_current -= __n; 
        return *this; 
    }
    
    __normal_iterator operator-(difference_type __n) const 
    { 
        return __normal_iterator(_M_current - __n); 
    }

    const _Iterator& base() const 
    { return _M_current; }
};






template<typename Iter>
size_t distance_iter(Iter first, Iter last)
{
    size_t result{};
    for (;first != last;++first)
        ++result;
    return result;

}


int main() {
    vector values = { 3, 8, 1, 4, 9 };

    vector<int>::iterator iter=values.begin();

    cout<<distance_iter(values.begin(), values.end())<<endl;

    sort(values.begin(), values.end(), greater<int>{});

    for_each(values.begin(), values.end(), [](int data){cout<<data<<" ";});

    const type_info& type= typeid(values.begin());
    cout<<type.name()<<endl;
}