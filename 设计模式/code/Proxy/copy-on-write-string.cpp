
class string
{
    using LongString = vector<char>;

    shared_ptr<LongString>      m_buffer;
    size_t                  m_length;

    void do_copy()
    {
        if( m_buffer.use_count() > 1 )
        {
            m_buffer = make_shared<LongString>( *m_buffer );
        }
    }

public:
    const char* c_str() const 
    { return m_buffer->data(); }

    size_t length() const 
    { return m_length; }

    const char& operator[]( const size_t i ) const 
    { return (*m_buffer)[i]; }

    char& operator[]( const size_t i )
    {
        do_copy();
        return (*m_buffer)[i];
    }
    
    template< Size n >
    string( const char* literal,size_t n ):
        m_buffer( make_shared<LongString>( literal, literal + n ) ),
        m_length( n - 1 )
    {}
};
