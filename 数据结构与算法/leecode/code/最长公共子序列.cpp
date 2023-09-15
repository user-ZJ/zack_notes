class Solution {
public:
    /**
     * longest common subsequence
     * @param s1 string字符串 the string
     * @param s2 string字符串 the string
     * @return string字符串
     */
    // if s1[i]==s2[j] dp[i][j] = dp[i-1][j-1] + 1
    // else dp[i][j] = max(dp[i-1][j],dp[i][j-1])
    string LCS(string s1, string s2) {
        // write code here
        if(s1.length()==0 || s2.length()==0)
            return "-1";
        int m=s1.length(),n=s2.length();
        int dp[m+1][n+1];
        for(int i=0;i<=m;i++) dp[i][0] = 0;
        for(int j=0;j<=n;j++) dp[0][j] = 0;
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(s1[i-1]==s2[j-1]){
                    dp[i][j] = dp[i-1][j-1] +1;
                }else{
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        //反向推结果
        int i=m,j=n;
        string res="";
        while(i>0&&j>0){
            if(s1[i-1]==s2[j-1]){   //对应dp[i][j] = dp[i-1][j-1] +1;的反推
                res = s1[i-1]+res;
                i--;
                j--;
            }else{   //对应dp[i][j] = max(dp[i-1][j],dp[i][j-1]);的反推
                if(dp[i-1][j] < dp[i][j-1]){
                    j--;
                }else if(dp[i-1][j] > dp[i][j-1]){
                    i--;
                }else if(dp[i-1][j] == dp[i][j-1]){
                    j--;
                }
            }
        }
        return res==""?"-1":res;
    }
};
