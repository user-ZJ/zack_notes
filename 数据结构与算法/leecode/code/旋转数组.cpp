#include <string>
#include <vector>
class Solution {
public:
  void rotate(std::vector<int> &nums, int k) {
    if (k > 0) {
      k = k % nums.size();
      k = nums.size() - k;
      rotateRight(nums.begin(), nums.begin() + k - 1);
      rotateRight(nums.begin() + k, nums.begin() + nums.size() - 1);
      rotateRight(nums.begin(), nums.begin() + nums.size() - 1);
    }
  }

  void rotateRight(std::vector<int>::iterator begin,
                   std::vector<int>::iterator end) {
    while (begin < end) {
      std::swap(*begin, *end);
      ++begin;
      --end;
    }
  }

  std::string LeftRotateString(std::string str, int n) {
    if (!str.empty()) {
      int nLength = str.length();
      if (nLength > 0 && n > 0 && n < nLength) {
        Reverse(str, 0, n - 1);
        Reverse(str, n, nLength - 1);
        Reverse(str, 0, nLength - 1);
      }
    }
    return str;
  }
  void Reverse(std::string &str, int begin, int end) {
    while (begin < end) {
      std::swap(str[begin], str[end]);
      begin++;
      end--;
    }
  }
};