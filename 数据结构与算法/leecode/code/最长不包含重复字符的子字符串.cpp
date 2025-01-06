class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> occ;
        int maxlen = 0;
        int i=0,j=0;
        while(j<s.size()){
            if(occ.count(s[j])==0){
                occ.insert(s[j]);
                maxlen = max(maxlen,j-i+1);
                ++j;
            }else{
                occ.erase(s[i]);
                ++i;
            }
        }
        return maxlen;
    }
};