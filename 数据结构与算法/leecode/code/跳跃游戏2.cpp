#include <vector>
class Solution {
public:
  int jump(std::vector<int> &nums) {
    int maxPos = 0, end = 0;
    int N = nums.size(), step = 0;
    for (int i = 0; i < N - 1; ++i) {
      if (maxPos >= i) {
        maxPos = std::max(maxPos, i + nums[i]);
        if (i == end) {
          end = maxPos;
          ++step;
        }
      }
    }
    return step;
  }
};