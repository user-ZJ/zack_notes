
template <class _Tp, class _Container = deque<_Tp> > 
class stack;

template <class _Tp, class _Container>
bool operator==(const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y);

template <class _Tp, class _Container>
bool operator< (const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y);

template <class _Tp, class _Container>
class  stack
{
public:
    typedef _Container                               container_type;
    typedef typename container_type::value_type      value_type;
    typedef typename container_type::reference       reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::size_type       size_type;
    static_assert((is_same<_Tp, value_type>::value), "" );

protected:
    container_type c;

public:
    
    stack(): c() {}

    stack(const stack& __q) : c(__q.c) {}

    stack& operator=(const stack& __q) {c = __q.c; return *this;}

    stack(stack&& __q): c(std::move(__q.c)) {}

    stack& operator=(stack&& __q)
    {
        c = std::move(__q.c); 
        return *this;
    }

    explicit stack(container_type&& __c) : c(std::move(__c)) {}

    explicit stack(const container_type& __c) : c(__c) {}

    template <class _Alloc>
    explicit stack(const _Alloc& __a,
                       _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(__a) {}

    template <class _Alloc>
        stack(const container_type& __c, const _Alloc& __a,
              _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(__c, __a) {}

    template <class _Alloc>
        stack(const stack& __s, const _Alloc& __a,
              _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(__s.c, __a) {}


    template <class _Alloc>
        stack(container_type&& __c, const _Alloc& __a,
              _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(std::move(__c), __a) {}

    template <class _Alloc>
        stack(stack&& __s, const _Alloc& __a,
              _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(std::move(__s.c), __a) {}

    bool empty()     const      {return c.empty();}
    
    size_type size() const      {return c.size();}
    
    reference top()             {return c.back();}
    
    const_reference top() const {return c.back();}

    void push(const value_type& __v) {c.push_back(__v);}
    
    void push(value_type&& __v) {c.push_back(std::move(__v));}

    template <class... _Args>
    decltype(auto) emplace(_Args&&... __args)
    { 
        return c.emplace_back(std::forward<_Args>(__args)...);
    }

    void pop() {c.pop_back();}

    
    void swap(stack& __s)
    {
        using std::swap;
        swap(c, __s.c);
    }

    template <class T1, class _C1>
    friend bool operator==(const stack<T1, _C1>& __x, const stack<T1, _C1>& __y);

    template <class T1, class _C1>
    friend bool operator< (const stack<T1, _C1>& __x, const stack<T1, _C1>& __y);
};


template<class _Container,class = _EnableIf<!__is_allocator<_Container>::value>>
stack(_Container) -> stack<typename _Container::value_type, _Container>;

template<class _Container,
         class _Alloc,
         class = _EnableIf<!__is_allocator<_Container>::value>,
         class = _EnableIf<uses_allocator<_Container, _Alloc>::value>
         >
stack(_Container, _Alloc)
    -> stack<typename _Container::value_type, _Container>;


template <class _Tp, class _Container>
inline bool
operator==(const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y)
{
    return __x.c == __y.c;
}

template <class _Tp, class _Container>
inline bool
operator< (const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y)
{
    return __x.c < __y.c;
}

template <class _Tp, class _Container>
inline bool
operator!=(const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y)
{
    return !(__x == __y);
}

template <class _Tp, class _Container>
inline bool
operator> (const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y)
{
    return __y < __x;
}

template <class _Tp, class _Container>
inline bool operator>=(const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y)
{
    return !(__x < __y);
}

template <class _Tp, class _Container>
inline bool operator<=(const stack<_Tp, _Container>& __x, const stack<_Tp, _Container>& __y)
{
    return !(__y < __x);
}

template <class _Tp, class _Container>
inline _EnableIf<__is_swappable<_Container>::value, void>
swap(stack<_Tp, _Container>& __x, stack<_Tp, _Container>& __y)
{
    __x.swap(__y);
}

template <class _Tp, class _Container, class _Alloc>
struct  uses_allocator<stack<_Tp, _Container>, _Alloc>
    : public uses_allocator<_Container, _Alloc>
{
};


