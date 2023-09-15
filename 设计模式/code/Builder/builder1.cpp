
struct Header{ };
struct Page{ };
struct Body{
    vector<Page> pages;
};
struct Footer{  };

class Document
{
    Header header;
    Body body;
    Footer footer;

protected:
    virtual Header buildHeader()=0;
    virtual Body buildBody()=0;
    virtual Page buildPage(int index)=0;
    virtual Footer buildFooter()=0;

public: 

    Document()
    {
        header= buildHeader(); // 静态绑定

        int pageNumber=readPageNumber(); 
        body = buildBody(); // 静态绑定

        for (int i=0;i<pageNumber;i++)
        {
            body.pages[i]=buildPage(i);// 静态绑定
        }
        footer=buildFooter();// 静态绑定

        return doc;
    }
    

};


class HTMLDocument: public Document
{
    int data;
public:
    HTMLDocument():Document(){

        //..... 初始化HTMLDocument
        data=100;
    }
protected:
     Header buildHeader() override {
        //
        cout<<data<<endl;
     }
    Body buildBody() override {

    }
    Page buildPage(int index) override {

    }
    Footer buildFooter() override{

    }

};

class MarkdownDocument: public Document
{
protected:
     Header buildHeader() override {

     }
    Body buildBody() override {

    }
    Page buildPage(int index) override {

    }
    Footer buildFooter() override{

    }

};