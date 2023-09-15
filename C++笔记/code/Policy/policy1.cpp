#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
#include <list>
using namespace std;

//Policy-based Design 

class MarkdownListPolicy
{
public:
  void start(ostringstream& oss) {}
  void end(ostringstream& oss) {}
  void add_list_item(ostringstream& oss, const string& item) 
  {
    oss << " * " << item << endl;
  }
};

class HtmlListPolicy
{
public:
  void start(ostringstream& oss) 
  {
    oss << "<ul>" << endl;
  }

  void end(ostringstream& oss) 
  {
    oss << "</ul>" << endl;
  }

  void add_list_item(ostringstream& oss, const string& item) 
  {
    oss << "<li>" << item << "</li>" << endl;
  }
};



template <typename ListPolicy>
class TextProcessor
{
public:
  void clear()
  {
    oss.str("");
    oss.clear();
  }
  void append_list(const vector<string> items)
  {
    list_policy.start(oss);
    for (auto& item : items)
      list_policy.add_list_item(oss, item);
    list_policy.end(oss);
  }

  string str() const { return oss.str(); }
private:
  ostringstream oss;
  ListPolicy list_policy;
};



int main()
{



  // markdown
  TextProcessor<MarkdownListPolicy> tp1;
  tp1.append_list({"Policy", "Design", "Pattern"});
  cout << tp1.str() << endl;
  cout<<sizeof(tp1)<<endl;

   // html
  TextProcessor<HtmlListPolicy> tp2;
  tp2.append_list({"Policy", "Design", "Pattern"});
  cout << tp2.str() << endl;
  cout<<sizeof(tp2)<<endl;

  getchar();
  return 0;
}