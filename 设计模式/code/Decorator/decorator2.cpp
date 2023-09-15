//业务操作
class Stream{
     
public：
    virtual char Read(int number)=0;
    virtual void Seek(int position)=0;
    virtual void Write(char data)=0;
    virtual ~Stream(){}
};

// 1+n+m

//主体类
class FileStream: public Stream{
public:
    char Read(int number) override {
        //读文件流
    }
    void Seek(int position) override{
        //定位文件流
    }
     void Write(char data) override{
        //写文件流
    }

};

class NetworkStream :public Stream{
public:
     char Read(int number) override{
        //读网络流
    }
     void Seek(int position) override{
        //定位网络流
    }
     void Write(char data) override{
        //写网络流
    }
    
};

class MemoryStream :public Stream{
public:
     char Read(int number) override{
        //读内存流
    }
     void Seek(int position) override{
        //定位内存流
    }
     void Write(char data) override{
        //写内存流
    }
    
};

//扩展操作
class CryptoStream: public Stream {

    unique_ptr<Stream> ps;// =new FileSream, new Networkstream, ...
public:
    CryptoStream(unique_ptr<Stream> s):ps( std::move(s))
    {

    }

     char Read(int number) override{
       
        //额外的加密操作...

        ps->Read(number);//读文件流
        
        //额外的加密操作...

    }
     void Seek(int position) override{
        //额外的加密操作...
        ps->Seek(position);//定位文件流
        //额外的加密操作...
    }
     void Write(byte data)override{
        //额外的加密操作...
        
        ps->Write(data);//写文件流
        //额外的加密操作...
    }
};


class BufferedStream : public Stream{
    unique_ptr<Stream> ps;// =new FileSream, new Networkstream, ...
public:
    BufferedStream(unique_ptr<Stream> s):ps( std::move(s))
    {

    }

    //...
};



void Process(){

    //编译时装配
    //CryptoFileStream *fs1 = new CryptoFileStream();

    //BufferedFileStream *fs2 = new BufferedFileStream();

    //CryptoBufferedFileStream *fs3 =new CryptoBufferedFileStream();

    unique_ptr<Stream> ps=make_unique<FileSteam>();
    CryptoStream cs(ps);

}