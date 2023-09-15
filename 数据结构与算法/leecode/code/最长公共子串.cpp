class Solution {
public:
    /**
     * longest common substring
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    string LCS(string str1, string str2) {
        // write code here
        int row = str1.length()+1;
        int col = str2.length()+1;
        int dp[row][col];
        int maxlen = 0,end=0;
        for(int i=0;i<row;i++) dp[i][0] = 0;
        for(int j=0;j<col;j++) dp[0][j] = 0;
        for(int i=1;i<row;i++){
            for(int j=1;j<col;j++){
                if(str1[i-1]==str2[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
                else dp[i][j] = 0;
                if(dp[i][j] > maxlen){
                    maxlen = dp[i][j];
                    end = j-1;
                }
            }
        }
        string res;
        if(maxlen==0) return "-1";
        else res = str2.substr(end-maxlen+1,maxlen);
        return res;
    }
};