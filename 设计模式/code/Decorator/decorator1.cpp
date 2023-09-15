//业务操作
class Stream{
public：
    virtual char Read(int number)=0;
    virtual void Seek(int position)=0;
    virtual void Write(char data)=0;
    virtual ~Stream(){}
};

/*
1 : Stream 

n=3 : FileStream, MemoryStream, NetworkStream 主体类

m=2 : Crypto, Buffered 扩展类

3* 2+ 3* 1= 9

9+3+1= 13 个类


1+n+ n*( m+ m-1+ m-2 + ....1)


n=5, m=5
=1+5+ 5*(5+....+1)= 1+5+5*15= 81
*/



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
class CryptoFileStream :public FileStream{
public:
     char Read(int number) override{
       
        //额外的加密操作...

        FileStream::Read(number);//读文件流
        
        //额外的加密操作...

    }
     void Seek(int position) override{
        //额外的加密操作...
        FileStream::Seek(position);//定位文件流
        //额外的加密操作...
    }
     void Write(byte data)override{
        //额外的加密操作...
        
        FileStream::Write(data);//写文件流
        //额外的加密操作...
    }
};

class CryptoNetworkStream : public NetworkStream{
public:
     char Read(int number)override{
        
        //额外的加密操作...

        NetworkStream::Read(number);//读网络流

         //额外的加密操作...
    }
     void Seek(int position)override{
        //额外的加密操作...
        NetworkStream::Seek(position);//定位网络流
        //额外的加密操作...
    }
     void Write(byte data)override{
        //额外的加密操作...
        NetworkStream::Write(data);//写网络流
        //额外的加密操作...
    }
};

class CryptoMemoryStream : public MemoryStream{
public:
     char Read(int number) override{
        
        //额外的加密操作...
        MemoryStream::Read(number);//读内存流
         //额外的加密操作...
    }
     void Seek(int position)override{
        //额外的加密操作...
        MemoryStream::Seek(position);//定位内存流
        //额外的加密操作...
    }
     void Write(byte data)override{
        //额外的加密操作...
        MemoryStream::Write(data);//写内存流
        //额外的加密操作...
    }
};


class BufferedFileStream : public FileStream{
    //...
};

class BufferedNetworkStream : public NetworkStream{
    //...
};

class BufferedMemoryStream : public MemoryStream{
    //...
};




class CryptoBufferedFileStream :public BufferedFileStream{
public:
     char Read(int number)override{
        
        //额外的加密操作...

        BufferedFileStream::Read(number);//读文件流
    }
     void Seek(int position)override{
        //额外的加密操作...
      
        BufferedFileStream::Seek(position);//定位文件流
        //额外的加密操作...
        
    }
     void Write(byte data)override{
        //额外的加密操作...

        BufferedFileStream::Write(data);//写文件流
        //额外的加密操作...
  
    }
};

class CryptoBufferedNetworkStream :public BufferedNetworkStream {

    //...
};

class CryptoBufferedMemoryStream :public BufferedMemoryStream {
    //...
};


void Process(){

    //编译时装配

    Stream * fs2=new FileStream();

    Stream *fs1 = new CryptoFileStream();

    Stream *fs2 = new BufferedFileStream();

    Stream *fs3 =new CryptoBufferedFileStream();

    delete fs1;
    delete fs2;
    delete fs3;

}