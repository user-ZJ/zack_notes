#include <iostream>
using namespace std;




class Stream  {
 public:
    virtual void Read(int number)
    {
          cout<<"Stream.Read()"<<endl;
    }

    virtual void Seek(int position)
    {
          cout<<"Stream.Seek()"<<endl;
    }
    virtual void Write(char data)
    {
        cout<<"Stream.Write()"<<endl;
    }
};

//1+m+n

class FileStream: public Stream{
public:



    void Read(int number) override
    {
          cout<<"FileStream.Read()"<<endl;
    }

    void Seek(int position) override
    {
          cout<<"FileStream.Seek()"<<endl;
    }
    void Write(char data) override
    {
        cout<<"FileStream.Write()"<<endl;
    }

    void close()
    {

    }

};

class NetworkStream :public Stream{
public:

     void Read(int number) override
    {
          cout<<"NetworkStream.Read()"<<endl;
    }

    void Seek(int position) override
    {
          cout<<"NetworkStream.Seek()"<<endl;
    }
    void Write(char data) override
    {
        cout<<"NetworkStream.Write()"<<endl;
    }

};

class MemoryStream :public Stream{
public:
    

    void Read(int number) override
    {
          cout<<"MemoryStream.Read()"<<endl;
    }

    void Seek(int position) override
    {
          cout<<"MemoryStream.Seek()"<<endl;
    }
    void Write(char data) override
    {
        cout<<"MemoryStream.Write()"<<endl;
    }
    
};



template<typename StreamType>
class CryptoStream : public StreamType{
public:

     void Read(int number) override  {
       
        cout<<"Read 加密..."<<endl;

        StreamType::Read(number);
    }
     void Seek(int position)  override  {

        cout<<"Seek 加密..."<<endl;

        StreamType::Seek(position);
        
    }
     void Write(char data)  override {
        
        cout<<"Write 加密..."<<endl;

        StreamType::Write(data);

    }
};

template<typename StreamType>
class BufferedStream :public StreamType{
public:
     void Read(int number) override  {
       
        cout<<"Read 缓存..."<<endl;

        StreamType::Read(number);
    }
     void Seek(int position) override  {

        cout<<"Seek 缓存..."<<endl;

        StreamType::Seek(position);
        
    }
     void Write(char data) override  {
        
        cout<<"Write 缓存..."<<endl;

        StreamType::Write(data);

    }
};

int main()
{
    CryptoStream<BufferedStream<NetworkStream>> stream;

    stream.Read(300);
    stream.Seek(100);
    stream.Write('a');

    


}