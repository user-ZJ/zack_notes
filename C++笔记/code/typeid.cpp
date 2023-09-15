class Test{};

int main(int argc, char *argv[]) {
  int i;
  const std::type_info &info1 = typeid(i);
  std::cout<<info1.name()<<std::endl;

  double j;
  const std::type_info &info2 = typeid(j);
  std::cout<<info2.name()<<std::endl;

  Test t;
  const std::type_info &info3 = typeid(t);
  std::cout<<info3.name()<<std::endl;

  Test *t1;
  const std::type_info &info4 = typeid(t1);
  std::cout<<info4.name()<<std::endl;
}