#include <algorithm>
#include <iterator>
#include <list>
#include <vector>
#include <stack>
#include <queue>
#include <iostream>
using namespace std;

int main()
{
    //stack<int> stk0;
    //stack<int,deque<int>> stk1;
    stack<int,vector<int>> stk1;
    stk1.push(2);
    stk1.push(4);
    stk1.push(6);
    stk1.push(8);
    stk1.push(10);

    stk1.pop();
    cout<<stk1.top()<<endl;
    cout<<stk1.size()<<endl;

    //queue<int> que1;
    queue<int, vector<int>> que1;

    que1.push(2);
    que1.push(4);
    que1.push(6);

    que1.pop();
    que1.front();
    que1.back();


}