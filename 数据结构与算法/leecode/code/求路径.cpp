class Solution {
public:
    /**
     * 
     * @param m int整型 
     * @param n int整型 
     * @return int整型
     */
    // dp[i][j] == dp[i-1][j] + dp[i][j-1]
    int uniquePaths(int m, int n) {
        // write code here
        if(m==0 || n==0) return 0;
        if(m==1 || n==1) return 1;
        int dp[m][n];
        int i,j;
        for(i=0;i<m;i++){
            dp[i][0] = 1;
        }  
        for(j=0;j<n;j++) {
            dp[0][j] = 1;
        }
        for(i=1;i<m;i++){
            for(j=1;j<n;j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};