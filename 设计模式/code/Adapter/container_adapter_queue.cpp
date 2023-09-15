template <class _Tp, class _Container = deque<_Tp> > 
class  queue;

template <class _Tp, class _Container>
bool operator==(const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y);

template <class _Tp, class _Container>
bool operator< (const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y);

template <class _Tp, class _Container /*= deque<_Tp>*/>
class  queue
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
    
    queue()
        _NOEXCEPT_(is_nothrow_default_constructible<container_type>::value)
        : c() {}

    
    queue(const queue& __q) : c(__q.c) {}

    
    queue& operator=(const queue& __q) {c = __q.c; return *this;}


    queue(queue&& __q)
        _NOEXCEPT_(is_nothrow_move_constructible<container_type>::value)
        : c(std::move(__q.c)) {}

    
    queue& operator=(queue&& __q)
        _NOEXCEPT_(is_nothrow_move_assignable<container_type>::value)
        {c = std::move(__q.c); return *this;}

    
    explicit queue(const container_type& __c)  : c(__c) {}
    
    explicit queue(container_type&& __c) : c(std::move(__c)) {}

    template <class _Alloc>
        explicit queue(const _Alloc& __a,
                       _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(__a) {}

    template <class _Alloc>
        queue(const queue& __q, const _Alloc& __a,
                       _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(__q.c, __a) {}

    template <class _Alloc>       
        queue(const container_type& __c, const _Alloc& __a,
                       _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(__c, __a) {}

    template <class _Alloc>
        queue(container_type&& __c, const _Alloc& __a,
                       _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(std::move(__c), __a) {}

    template <class _Alloc>
        queue(queue&& __q, const _Alloc& __a,
                       _EnableIf<uses_allocator<container_type, _Alloc>::value>* = 0)
            : c(std::move(__q.c), __a) {}

    bool      empty() const {return c.empty();}
    
    size_type size() const  {return c.size();}

    reference       front()       {return c.front();}
    
    const_reference front() const {return c.front();}
    
    reference       back()        {return c.back();}
    
    const_reference back() const  {return c.back();}
    
    void push(const value_type& __v) {c.push_back(__v);}
    
    void push(value_type&& __v)      {c.push_back(std::move(__v));}
 
    template <class... _Args>
        decltype(auto) emplace(_Args&&... __args)
            { return c.emplace_back(std::forward<_Args>(__args)...);}
    
    void pop() {c.pop_front();}

    
    void swap(queue& __q)
    {
        using std::swap;
        swap(c, __q.c);
    }

    template <class _T1, class _C1>
    friend bool operator==(const queue<_T1, _C1>& __x,const queue<_T1, _C1>& __y);

    template <class _T1, class _C1>
    friend bool operator< (const queue<_T1, _C1>& __x,const queue<_T1, _C1>& __y);
};


template<class _Container,
         class = _EnableIf<!__is_allocator<_Container>::value>>
queue(_Container)-> queue<typename _Container::value_type, _Container>;



template <class _Tp, class _Container>
inline bool operator==(const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y)
{
    return __x.c == __y.c;
}

template <class _Tp, class _Container>
inline bool operator< (const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y)
{
    return __x.c < __y.c;
}

template <class _Tp, class _Container>
inline bool operator!=(const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y)
{
    return !(__x == __y);
}

template <class _Tp, class _Container>
inline bool operator> (const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y)
{
    return __y < __x;
}

template <class _Tp, class _Container>
inline bool operator>=(const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y)
{
    return !(__x < __y);
}

template <class _Tp, class _Container>
inline bool operator<=(const queue<_Tp, _Container>& __x,const queue<_Tp, _Container>& __y)
{
    return !(__y < __x);
}

template <class _Tp, class _Container>
inline  _EnableIf<__is_swappable<_Container>::value, void>
swap(queue<_Tp, _Container>& __x, queue<_Tp, _Container>& __y)
{
    __x.swap(__y);
}

