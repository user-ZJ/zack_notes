#include <algorithm>
#include <iterator>
#include <list>
#include <vector>
#include <deque>
#include <set>
#include <iostream>
using namespace std;

int main()
{
      string s;
      
      list<int> coll1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

      vector<int> coll2;

      //back_insert_iterator
      copy (coll1.cbegin(), coll1.cend(),    
            back_inserter(coll2)); 

      for_each(coll2.begin(), coll2.end(),[](const auto& elem){ cout<<elem<<" ";});          


      //front_insert_iterator
      deque<int> coll3;
      copy (coll1.cbegin(), coll1.cend(),    
            front_inserter(coll3));          


      //insert_iterator
      set<int> coll4;
      copy (coll1.cbegin(), coll1.cend(),    
            inserter(coll4,coll4.begin()));  
}


