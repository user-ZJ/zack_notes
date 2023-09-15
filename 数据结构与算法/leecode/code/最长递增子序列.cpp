class Solution {
public:
    /**
     * retrun the longest increasing subsequence
     * @param arr int整型vector the array
     * @return int整型vector
     */
    vector<int> LIS(vector<int>& arr) {
        // write code here
        // 第一步：利用贪心+二分求最长递增子序列长度
        vector<int> res;
        vector<int> maxLen;
        if (arr.size() < 1) return res;
        res.push_back(arr[0]);  // 存放最终序列
        maxLen.push_back(1);  // 存放arr[i]为尾元素的最长子序列的长度
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[i] > res.back()) {
                res.emplace_back(arr[i]);
                maxLen.emplace_back(res.size());
            } else {
                // lower_bound(begin, end, val)包含在<algorithm>中
                // 它的作用是返回有序数组begin..end中第一个大于等于val的元素的迭代器
                int pos = lower_bound(res.begin(), res.end(), arr[i]) - res.begin();
                res[pos] = arr[i];
                maxLen.emplace_back(pos+1);
            }
        }
        // 第二步：填充最长递增子序列
        for (int i = arr.size()-1, j = res.size(); j > 0; --i) {
            if (maxLen[i] == j) {
                res[--j] = arr[i];
            }
        }
        return res;
    }
};