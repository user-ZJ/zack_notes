class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n=nums.size();
        if(n<=1)
            return n;
        int slow=1,fast=1;
        while(fast<n){
            if(nums[fast]!=nums[fast-1]){
                nums[slow++] = nums[fast];
            }
            ++fast;
        }
        return slow;
    }
};