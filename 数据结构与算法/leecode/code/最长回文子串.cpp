class Solution {
public:
    /*
    动态规划
    dp[i][j]是一个bool类型的变量数组，如果dp[i][j]==true,
    那么他表示字符串str从str[i]到str[j]是回文串

    边界是：
        dp[i][i]=true,dp[i][i+1]=(str[i]==str[i+1]) ? true , false
    状态转移方程：
        dp[i][j]=true if( dp[i+1][j-1] && str[i]==str[j] )
        dp[i][j]=false if( str[i]!=str[j] )
    */
    int getLongestPalindrome(string A, int n) {
        // write code here
        bool dp[n][n];
        string res="";
        ////l:字符串首尾字母长度差 (d = j-i)
        for(int l=0;l<n;++l){
            for(int i=0;i+l<n;++i){ // 字符串起始位置 i
                int j = i + l; // 字符串结束位置 j
                if(l==0){
                    dp[i][j] = true;
                }else if(l==1){
                    dp[i][j] = (A[i]==A[j]);
                }else{
                    dp[i][j] = (A[i]==A[j] && dp[i+1][j-1]);
                }
                if(dp[i][j] && l+1>res.length()){
                    res = A.substr(i,l+1);
                }
            }
        }
        return res.length();
    }
};