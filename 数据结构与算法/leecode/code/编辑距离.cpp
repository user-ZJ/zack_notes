int editDistance(const std::string &src, const std::string &tgt) {
  std::wstring wsrc = to_wstring(src), wtgt = to_wstring(tgt);
  int m = wsrc.size(), n = wtgt.size();
  int dp[m + 1][n + 1];
  for (int i = 0; i <= m; ++i)
    dp[i][0] = i;
  for (int j = 0; j <= n; ++j)
    dp[0][j] = j;
  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      if (wsrc[i] == wtgt[j])
        dp[i][j] = dp[i - 1][j - 1];
      else {
        dp[i][j] =
            std::min(dp[i - 1][j], std::min(dp[i][j - 1], dp[i - 1][j - 1])) +
            1;
      }
    }
  }
  return dp[m][n];
}