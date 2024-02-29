class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int p1 = m-1,p2=n-1;
        int tail = m+n-1;
        while(p1>=0 and p2>=0){
            if(nums1[p1]>nums2[p2]){
                nums1[tail--] = nums1[p1--];
            }else{
                nums1[tail--] = nums2[p2--];
            }
        }
        while(p1>=0 and tail>=0){
            nums1[tail--] = nums1[p1--];
        }
        while(p2>=0 and tail>=0){
            nums1[tail--] = nums2[p2--];
        }
    }
};