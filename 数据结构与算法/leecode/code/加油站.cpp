class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int len = gas.size();
        int spare = 0;
        int minSpare = 1e9;
        int minIndex = 0;
        for(int i=0;i<len;++i){
            spare += gas[i] - cost[i];
            if(spare < minSpare){
                minSpare = spare;
                minIndex = i;
            }
        }
        if(spare<0){
            return -1;
        }else{
            return (minIndex+1)%len;
        }
    }
};