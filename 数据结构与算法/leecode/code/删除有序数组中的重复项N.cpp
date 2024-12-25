class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int len=0,k=2;
        for(auto &num:nums){
            if(len<k || nums[len-k]!=num){
                nums[len++] = num;
            }
        }
        return len;
    }
};