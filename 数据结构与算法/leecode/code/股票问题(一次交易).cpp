#include <vector>
class Solution {
public:
  int maxProfit(std::vector<int> &prices) {
    int inf = 1e9;
    int minprice = inf, maxprofit = 0;
    for (int price : prices) {
      maxprofit = std::max(maxprofit, price - minprice);
      minprice = std::min(price, minprice);
    }
    return maxprofit;
  }

  int maxProfit1(std::vector<int> &prices) {
    // write code here
    int N = prices.size();
    int dp[N + 1][2];
    dp[0][0] = 0;
    dp[0][1] = std::INT_MIN;
    for (int i = 1; i < N + 1; i++) {
      dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1]);
      dp[i][1] = std::max(dp[i - 1][1], -prices[i - 1]);
    }
    return dp[N][0];
  }
};