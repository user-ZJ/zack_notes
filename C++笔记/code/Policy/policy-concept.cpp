#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
#include <list>
using namespace std;

template <typename T>
concept ListPolicy = 
requires(T t, const string& item) {
  t.list_item(item);
  t.start();
  t.end();
  t.get_text();
};

class MarkdownListStrategy {
  ostringstream oss;
public:
   void start() { oss << "*" << endl;}
   void end() { oss << "*" << endl;}
   void list_item(const string& item) { oss << " * * " << item << endl;} 
   string get_text() const { return oss.str(); }
};

static_assert(ListPolicy<MarkdownListStrategy>);


class HtmlListStrategy
{
  ostringstream oss;
public:

  void start() 
  {
    oss << "<ul>" << endl;
  }

  void end() 
  {
    oss << "</ul>" << endl;
  }

  void list_item(const string& item) 
  {
    oss << "<li>" << item << "</li>" << endl;
  }

  string get_text() const 
  { 
    return oss.str(); 
  }

};

template <ListPolicy ListType>
class TextProcessor : private  ListType 
{
public:
    void append_list(const vector<string> items)
    {
      ListType::start();
      for (auto& item : items)
        ListType::list_item(item);
      ListType::end();
    } 
    string getText() {return ListType::get_text();}
};

int main()
{
  // markdown
  TextProcessor<MarkdownListStrategy> tp1;
  tp1.append_list({"Concept", "Design", "Pattern"});
  cout << tp1.getText() << endl;
 
   // html
  TextProcessor<HtmlListStrategy> tp2;
  tp2.append_list({"Concept", "Design", "Pattern"});
  cout << tp2.getText() << endl;


  return 0;
}