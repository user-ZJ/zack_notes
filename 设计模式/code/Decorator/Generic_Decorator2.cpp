#include <iostream>
using namespace std;

template <typename T>
concept Stream = requires(T t, int value,char data) {
    t.Read(value);
    t.Seek(value);
    t.Write(data);
};



class FileStream{
public:
    void Read(int number) 
    {
          cout<<"FileStream.Read()"<<endl;
    }

    void Seek(int position) 
    {
          cout<<"FileStream.Seek()"<<endl;
    }
    void Write(char data) 
    {
        cout<<"FileStream.Write()"<<endl;
    }
};

static_assert(Stream<FileStream>);


class NetworkStream {
public:

     void Read(int number) 
    {
          cout<<"NetworkStream.Read()"<<endl;
    }

    void Seek(int position) 
    {
          cout<<"NetworkStream.Seek()"<<endl;
    }
    void Write(char data) 
    {
        cout<<"NetworkStream.Write()"<<endl;
    }

};
static_assert(Stream<NetworkStream>);



class MemoryStream{
public:
    

    void Read(int number) 
    {
          cout<<"MemoryStream.Read()"<<endl;
    }

    void Seek(int position) 
    {
          cout<<"MemoryStream.Seek()"<<endl;
    }
    void Write(char data) 
    {
        cout<<"MemoryStream.Write()"<<endl;
    }
    
};

static_assert(Stream<MemoryStream>);



template<Stream StreamType>
class CryptoStream : public StreamType{
public:


     void Read(int number)   {
       
        cout<<"Read 加密..."<<endl;

        StreamType::Read(number);
    }
     void Seek(int position)    {

        cout<<"Seek 加密..."<<endl;

        StreamType::Seek(position);
        
    }
     void Write(char data)   {
        
        cout<<"Write 加密..."<<endl;

        StreamType::Write(data);

    }
};




template<Stream StreamType>
class BufferedStream :public StreamType{
public:
     void Read(int number)   {
       
        cout<<"Read 缓存..."<<endl;

        StreamType::Read(number);
    }
     void Seek(int position)   {

        cout<<"Seek 缓存..."<<endl;

        StreamType::Seek(position);
        
    }
     void Write(char data)   {
        
        cout<<"Write 缓存..."<<endl;

        StreamType::Write(data);

    }
};

static_assert(Stream<CryptoStream<NetworkStream>>);
static_assert(Stream<BufferedStream<NetworkStream>>);



int main()
{
    CryptoStream<MemoryStream> stream;

    stream.Read(300);
    stream.Seek(100);
    stream.Write('a');


}