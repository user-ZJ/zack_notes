
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

   

    void buildDocument() // 构建流程不变
    {

        header= buildHeader(); //动态绑定

        int pageNumber=readPageNumber();
        body = buildBody();

        for (int i=0;i<pageNumber;i++)
        {
            body.pages[i]=buildPage(i);
        }
        footer=buildFooter();

        return doc;
    }
    

};


class HTMLDocument: public Document
{
protected:
    Header buildHeader() override {
        //######
    }
    Body buildBody() override {
        //######
    }
    Page buildPage(int index) override {
        //######
    }
    Footer buildFooter() override{
        //######
    }

};

class MarkdownDocument: public Document
{
protected:
    Header buildHeader() override {
        //@@@@@@
    }
    Body buildBody() override {
        //@@@@@@
    }
    Page buildPage(int index) override {
        //@@@@@@
    }
    Footer buildFooter() override{
        //@@@@@@
    }
};

int main(){

    MarkdownDocument doc{};

    doc.buildDocument();


}