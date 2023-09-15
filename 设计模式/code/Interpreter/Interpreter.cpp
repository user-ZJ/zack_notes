
#include <iostream>
#include <map>
#include <stack>

using namespace std;

class Expression {
public:
    virtual int interpreter(map<char, int> var)=0;
    virtual ~Expression(){}
};



//变量表达式
class VarExpression: public Expression {
    
    char key;
    
public:
    VarExpression(const char& key)
    {
        this->key = key;
    }
    
    int interpreter(map<char, int> var) override {
        return var[key];
    }
    
};

//符号表达式
class SymbolExpression : public Expression {

public: 
    // 运算符左右两个参数
    Expression* left;
    Expression* right;
    

    SymbolExpression( Expression* left,  Expression* right):
        left(left),right(right){
        
    }
    
};

//加法运算
class AddExpression : public SymbolExpression {
    
public:
    AddExpression(Expression* left, Expression* right):
        SymbolExpression(left,right){
        
    }
    int interpreter(map<char, int> var) override {
        return left->interpreter(var) + right->interpreter(var);
    }
    
};

//减法运算
class SubExpression : public SymbolExpression {
    
public:
    SubExpression(Expression* left, Expression* right):
        SymbolExpression(left,right){
        
    }
    int interpreter(map<char, int> var) override {
        return left->interpreter(var) - right->interpreter(var);
    }
    
};



Expression*  analyse(string expStr) {
    
    stack<Expression*> expStack;
    Expression* left = nullptr;
    Expression* right = nullptr;
    for(int i=0; i<expStr.size(); i++)
    {
        switch(expStr[i])
        {
            case '+':
                // 加法运算
                left = expStack.top();
                right = new VarExpression(expStr[++i]);
                expStack.push(new AddExpression(left, right));
                break;
            case '-':
                // 减法运算
                left = expStack.top();
                right = new VarExpression(expStr[++i]);
                expStack.push(new SubExpression(left, right));
                break;
            default:
                // 变量表达式
                expStack.push(new VarExpression(expStr[i]));
        }
    }
   
    Expression* expression = expStack.top();

    return expression;
}

//释放表达式树的节点内存...
void release(Expression* expression) {
    if(expression) {
        if(dynamic_cast<VarExpression*>(expression)) {
            delete expression;
        } else if(dynamic_cast<SymbolExpression*>(expression)) {
            SymbolExpression* symbolExp = dynamic_cast<SymbolExpression*>(expression);
            release(symbolExp->left);
            release(symbolExp->right);
            delete symbolExp;
        }
    }
}


int main() {
    
    
    string expStr = "a+b*c-d";// //   (a+ (b*c)) -d
    map<char, int> var;
    var.insert(make_pair('a',5));
    var.insert(make_pair('b',2));
    var.insert(make_pair('c',1));
    var.insert(make_pair('d',6));


    
    Expression* expression= analyse(expStr);
    
    int result=expression->interpreter(var);
    
    cout<<result<<endl;
    
    release(expression);
    
    return 0;
}

