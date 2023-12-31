#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
using namespace std;



class ListStrategy
{
public:
  virtual ~ListStrategy() = default;
  virtual void add_list_item(ostringstream& oss, const string& item)=0;
  virtual void start(ostringstream& oss)=0;
  virtual void end(ostringstream& oss) =0;
};



class MarkdownListStrategy : public ListStrategy
{
public:
  void add_list_item(ostringstream& oss, const string& item) override
  {
    oss << " * * " << item << endl;
  }

  void start(ostringstream& oss) override {
    oss << " * " << endl;
  }
  void end(ostringstream& oss) override {
    oss << " * " << endl;
  }
};

class HtmlListStrategy : public ListStrategy
{
public:
  void start(ostringstream& oss) override
  {
    oss << "<ul>" << endl;
  }

  void end(ostringstream& oss) override
  {
    oss << "</ul>" << endl;
  }

  void add_list_item(ostringstream& oss, const string& item) override
  {
    oss << "<li>" << item << "</li>" << endl;
  }
};

//扩展
struct XMLListStrategy :  ListStrategy
{
  void add_list_item(ostringstream& oss, const string& item) override {}
  void start(ostringstream& oss) override {}
  void end(ostringstream& oss) override {}
};

class TextProcessor
{
public:


  TextProcessor(unique_ptr<ListStrategy> ls):list_strategy(std::move(ls))
  {
   
  }
  void clear()
  {
    oss.str("");
    oss.clear();
  }

  //稳定--> 复用
  void append_list(const vector<string> items)
  {
    list_strategy->start(oss);
    for (auto& item : items)
      list_strategy->add_list_item(oss, item);
    list_strategy->end(oss);//动态辨析
  }


  string str() const { return oss.str(); }
private:
  ostringstream oss;
  
  //ListStrategy list_s; 错误！
  //ListStrategy* list_strategy;
  unique_ptr<ListStrategy> list_strategy;
};

int main()
{
  // markdown
  TextProcessor tp1(std::make_unique<MarkdownListStrategy>());
  tp1.append_list({"Software", "Design", "Pattern"});
  cout << tp1.str() << endl;


  TextProcessor tp2(std::make_unique<HtmlListStrategy>());
  tp2.append_list({"Software", "Design", "Pattern"});
  cout << tp2.str() << endl;

  getchar();
  return 0;
}