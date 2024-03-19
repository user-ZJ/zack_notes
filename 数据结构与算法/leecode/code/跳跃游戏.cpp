#include <vector>
class Solution {
public:
  bool canJump(std::vector<int> &nums) {
    int max_i = 0;
    for (int i = 0; i < nums.size(); i++) {
      // #如果当前位置能到达，并且当前位置+跳数>最远位置
      if (max_i >= i && i + nums[i] > max_i)
        max_i = i + nums[i];
    }
    return max_i >= nums.size() - 1;
  }
};