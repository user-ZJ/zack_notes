#include <vector>
int MoreThanHalfNum_Solution(std::vector<int> numbers) {
  if (numbers.size() == 0)
    return 0;
  int result = numbers[0];
  int times = 1;
  for (int i = 1; i < numbers.size(); i++) {
    if (times == 0) {
      result = numbers[i];
      times = 1;
    } else if (numbers[i] == result)
      ++times;
    else
      --times;
  }
  // check the result
  int count = 0;
  for (int i = 0; i < numbers.size(); i++) {
    if (numbers[i] == result)
      count++;
  }
  if (count * 2 <= numbers.size())
    return 0;
  return result;
}