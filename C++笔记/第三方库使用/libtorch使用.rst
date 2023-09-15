libtorch使用
=====================

下载
------------
https://pytorch.org/get-started/locally/


tensor操作
-------------------

.. code-block:: cpp

    #include <torch/torch.h>
    #include <iostream>
    // 使用固定维度，指定值初始化tensor
    torch::Tensor b = torch::zeros({3,4});
    b = torch::ones({3,4});
    b= torch::eye(4);  // 对角线全为1，其余全为0
    b = torch::full({3,4},10);
    b = torch::tensor({33,22,11});
    // 固定维度，随机值初始化
    torch::Tensor r = torch::rand({3,4});
    r = torch::randn({3, 4});
    r = torch::randint(0, 4,{3,3}); // 0-4范围的值初始化维度为3x3
    // C++中数据结构初始化tensor
    int aa[10] = {3,4,6};
    std::vector<float> aaaa = {3,4,6};
    auto aaaaa = torch::from_blob(aa,{3},torch::kFloat);
    auto aaa = torch::from_blob(aaaa.data(),{3},torch::kFloat);
    std::cout << aaa << std::endl;
    // 对已经创建的tensor进行初始化
    auto b = torch::zeros({3,4});
    auto d = torch::Tensor(b);
    d = torch::zeros_like(b);
    d = torch::ones_like(b);
    d = torch::rand_like(b,torch::kFloat);
    d = b.clone();
    // 存虚数的tensor
    torch::Tensor tt = torch::rand({2,3},torch::kComplexFloat);
    auto accessor = tt.accessor<c10::complex<float>, 2>();
    for (int i = 0; i < stft.size(0); i++) {
        for (int j = 0; j < stft.size(1); j++) {
            std::cout << accessor[i][j].real() << "+" << accessor[i][j].imag() << "i ";
        }
        std::cout << std::endl;
    }


    // 改变tensor的维度
    auto b = torch::full({10},3);
    b.view({1, 2,-1});
    std::cout<<b;
    b = b.view({1, 2,-1});
    std::cout<<b;
    auto c = b.transpose(0,1);
    std::cout<<c;
    auto d = b.reshape({1,1,-1});
    std::cout<<d;
    auto e = b.permute({1,0,2});
    std::cout<<e;

    // 切片
    auto b = torch::rand({10,3,28,28});//BxCxHxW
    std::cout<<b[0].sizes();//0th picture
    std::cout<<b[0][0].sizes();//0th picture, 0th channel
    std::cout<<b[0][0][0].sizes();//0th picture, 0th channel, 0th row pixels
    std::cout<<b[0][0][0][0].sizes();//0th picture, 0th channel, 0th row, 0th column pixels
    std::cout<<b.index_select(0,torch::tensor({0, 3, 3})).sizes();//choose 0th dimension at 0,3,3 to form a tensor of [3,3,28,28]
    std::cout<<b.index_select(1,torch::tensor({0,2})).sizes(); //choose 1th dimension at 0 and 2 to form a tensor of[10, 2, 28, 28]
    std::cout<<b.index_select(2,torch::arange(0,8)).sizes(); //choose all the pictures' first 8 rows [10, 3, 8, 28]
    std::cout<<b.narrow(1,0,2).sizes();//choose 1th dimension, from 0, cutting out a lenth of 2, [10, 2, 28, 28]
    std::cout<<b.select(3,2).sizes();//select the second tensor of the third dimension, that is, the tensor composed of the second row of all pictures [10, 3, 28]
    auto c = torch::randn({3,4});
    auto mask = torch::zeros({3,4});
    mask[0][0] = 1;
    std::cout<<c;
    std::cout<<c.index({mask.to(torch::kBool)});
    auto c = torch::randn({ 3,4 });
    auto mask = torch::zeros({ 3,4 });
    mask[0][0] = 1;
    mask[0][2] = 1;
    std::cout << c;
    std::cout << c.index({ mask.to(torch::kBool) });
    std::cout << c.index_put_({ mask.to(torch::kBool) }, c.index({ mask.to(torch::kBool) })+1.5);
    std::cout << c;

    // tensor操作
    auto b = torch::ones({3,4});
    auto c = torch::zeros({3,4});
    auto cat = torch::cat({b,c},1);//1 refers to 1th dim, output a tensor of shape [3,8]
    auto stack = torch::stack({b,c},1);//1refers to 1th dim, output a tensor of shape [3,2,4]
    std::cout<<b<<c<<cat<<stack;
    auto b = torch::rand({3,4});
    auto c = torch::rand({3,4});
    // mul div mm bmm
    std::cout<<b<<c<<b*c<<b/c<<b.mm(c.t());



