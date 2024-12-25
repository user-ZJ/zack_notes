class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> L(nums.size(),1),R(nums.size(),1);
        vector<int> answer(nums.size());
        for(int i=0;i<nums.size()-1;++i){
            L[i+1] = nums[i]*L[i];
        }
        for(int i=nums.size()-1;i>0;--i){
            R[i-1] = nums[i]*R[i];
        }
        for(int i=0;i<nums.size();++i)
            answer[i] = L[i]*R[i];
        return answer;
    }
};