#include <iostream>
#include <random>

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-10, 10);
    int random_num = dis(gen);
    std::cout << "随机数为：" << random_num << std::endl;

    std::vector<float> weights{0.1, 0.3, 0.6};
    // 创建分布
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    for(int i = 0; i < 10; i++) {
        std::cout<<dist(gen)<<std::endl;
    }
    return 0;
}