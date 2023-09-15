
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

    
};


//**************************

class DocumentBuilder{
public:

    virtual Header buildHeader()=0;
    virtual Body buildBody()=0;
    virtual Page buildPage(int index)=0;
    virtual Footer buildFooter()=0;

};

class Director{

    unique_ptr<DocumentBuilder> builder;

public:

    void setBuilder(unique_ptr<DocumentBuilder>  newBuilder)
    {
        builder = std::move(newBuilder);
    }

    unique_ptr<Document> buildDocument()  // 构建流程不变
    {
        
        unique_ptr<Document>  doc= make_unique<Document>();

        doc->header= builder->buildHeader();

        int pageNumber=readPageNumber();
        doc->body = builder->buildBody();

        for (int i=0;i<pageNumber;i++)
        {
            doc->body.pages[i]=builder->buildPage(i);
        }
        doc->footer=builder->buildFooter();

        return doc;
    }
    

};

class HtmlBuilder: public DocumentBuilder{
public:
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

class MarkdownBuilder: public DocumentBuilder{
public:
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