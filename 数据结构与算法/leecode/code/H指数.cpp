#include <vector>
class Solution {
public:
  int hIndex(std::vector<int> &citations) {
    int n = citations.size(), tot = 0;
    std::vector<int> counter(n + 1);
    for (int i = 0; i < n; i++) {
      if (citations[i] >= n) {
        counter[n]++;
      } else {
        counter[citations[i]]++;
      }
    }
    for (int i = n; i >= 0; i--) {
      tot += counter[i];
      if (tot >= i) {
        return i;
      }
    }
    return 0;
  }
};