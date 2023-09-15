
class Widget {

};

class WidgetCOW
{
    using  BigObject = vector<Widget>;

    shared_ptr<BigObject>      m_buffer;
    size_t                  m_length;

    void do_copy()
    {
        if( m_buffer.use_count() > 1 )
        {
            m_buffer = make_shared<BigObject>( *m_buffer );
        }
    }

public:

    WidgetCOW(const WidgetCOW&& )=default;
 
    size_t length() const 
    { return m_length; }

    const Widget& operator[]( const size_t i ) const 
    { return (*m_buffer)[i]; }

    //更改函数
    Widget& operator[]( const size_t i )
    {
        do_copy();
        return (*m_buffer)[i];
    }
    
};

int main(){

    WidgetCOW c1;
    WidgetCOW c2=c1;

    c2[0]=Widget{};
    

}
