# kaldi源码解析2-base

kaldi-base中包含io-funcs，kaldi-error，kaldi-math，kaldi-types，kaldi-utils，kaldi-common，timer，version。

## version

kaldi版本描述，通过get_version.sh脚本可以查看kaldi源码版本，get_version.sh脚本会去读src/.version文件，该文件中记录了kaldi的版本，当前版本为5.5

## kaldi-types

在kaldi-types.h中定义了再kaldi中使用的数据类型，包括：

**BaseFloat**：kaldi中用到最多的BaseFloat类型，如果定义了KALDI_DOUBLEPRECISION编译选项，BaseFloat使用双精度浮点类型，如果没有定义则使用单精度浮点类型。

```cpp
namespace kaldi {
// TYPEDEFS ..................................................................
#if (KALDI_DOUBLEPRECISION != 0)
typedef double  BaseFloat;
#else
typedef float   BaseFloat;
#endif
}
```

**基础数据类型**:在kaldi中基础数据类型引用的是fst中定义的数据类型

```cpp
#include <fst/types.h>
namespace kaldi {
  using ::int16;
  using ::int32;
  using ::int64;
  using ::uint16;
  using ::uint32;
  using ::uint64;
  typedef float   float32;
  typedef double double64;
}  // end namespace kaldi
```

查看fst/types.h可以发现原始定义为：

```cpp
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
```

## kaldi-error

kaldi-error是kaldi断言和日志系统的实现，包含KALDI_ASSERT, KALDI_ERR,KALDI_WARN, KALDI_LOG and KALDI_VLOG(v)

* KALDI_ASSERT:kaldi断言，如果断言错误，则退出程序
* KALDI_ERR：打印日志，并抛出异常
* KALDI_WARN：打印warning级别的日志
* KALDI_LOG：打印info级别的日志
* KALDI_VLOG(v)：自定义级别的日志，通过--verbose=?参数控制日志是否打印，最终是通过SetVerboseLevel(v)设置日志级别，例如--verbose=1，则KALDI_VLOG(0)和KALDI_VLOG(1)级别的日志都会被打印。verbose默认值为0

## kaldi-utils

CharToString：char转string

MachineIsLittleEndian：检查是不是小端机器

KALDI_SWAP8，KALDI_SWAP4，KALDI_SWAP2：内存交换

KALDI_DISALLOW_COPY_AND_ASSIGN：Makes copy constructor and operator= private

KALDI_ASSERT_IS_INTEGER_TYPE，KALDI_ASSERT_IS_FLOATING_TYPE：数据检查

KALDI_MEMALIGN：kaldi中内存申请

KALDI_MEMALIGN_FREE：kaldi中内存释放

## kaldi-math

**基本数值定义**：

DBL_EPSILON和 FLT_EPSILON：主要用于单精度和双精度的比较当中

M_PI和M_2PI：π和2π

M_SQRT2：2 的平方根

M_SQRT1_2：2 的平方根的倒数

M_LOG_2PI：log(2*PI)

M_LN2：ln(2)

M_LN10：ln(10)

**数学操作**：

Exp：exp操作

Log：log操作

Log1p：log(1 + x)

LogAdd：log(exp(x) + exp(y))

LogSub：log(exp(x) - exp(y))

ApproxEqual，AssertEqual：abs(a - b) <= relative_tolerance * (abs(a)+abs(b))

RandInt，RandUniform，RandGauss，RandPoisson，RandGauss2，RandPrune：生成随机数

RoundUpToNearestPowerOfTwo：返回一个不大于i的2的整数次幂

Gcd：最大公约数

Lcm：最小公倍数

Factorize：把一个数分解为质数

Hypot：计算直角三角形的斜边长

## io-funcs

kaldi的基本类型（整数和浮点类型以及bool）的输入输出，定义方式 WriteBasicType ReadBasicType，可以是二进制或文本模式，在kaldi中所有的输入和输出都是以流的形式体现，以方便kaldi各个bin之间的数据流转，还可以随时写入、读取磁盘文件。

InitKaldiInputStream(std::istream &is, bool *binary)：确定数据是文本还是二进制，主要判断是否包含\0B

kaldi二进制格式为

```text
index1 \0B<header><content>index2 \0B<header><content>
```

```cpp
inline bool InitKaldiInputStream(std::istream &is, bool *binary) {
  // Sets the 'binary' variable.
  // Throws exception in the very unusual situation that stream
  // starts with '\0' but not then 'B'.

  if (is.peek() == '\0') {  // seems to be binary
    is.get();
    if (is.peek() != 'B') {
      return false;
    }
    is.get();
    *binary = true;
    return true;
  } else {
    *binary = false;
    return true;
  }
}
```

InitKaldiOutputStream(std::ostream &os, bool binary):设置写入流是文本还是二进制，并设置数据精度

```cpp
inline void InitKaldiOutputStream(std::ostream &os, bool binary) {
  // 如果是二进制，写入\0B
  if (binary) {
    os.put('\0');
    os.put('B');
  }
  // 小数点后保留7位
  if (os.precision() < 7)
    os.precision(7);
}
```



## timer

kaldi内部计时器，可以用于程序运行时间统计等。

## kaldi-common

只有头文件，主要用于导入上面实现的内容





