class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        if(nums.empty())
            return 0;
        int left=0,right=0,sum=0;
        int len = nums.size()+1;
        while(right<nums.size()){
            sum += nums[right];
            while(sum>=target){
                len = min(len,right-left+1);
                sum -= nums[left];
                ++left;
            }
            ++right;
        }
        return len<nums.size()+1?len:0;
    }
};