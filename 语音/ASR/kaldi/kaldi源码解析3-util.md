# kaldi源码解析3-util

kaldi-util包含basic-filebuf，common-utils，const-integer-set，edit-distance，hash-list，kaldi-holder，kaldi-io，kaldi-pipebuf，kaldi-semaphore，kaldi-table，kaldi-thread，parse-options，simple-io-funcs，simple-options，stl-utils，table-types，text-utils

## basic-filebuf

对文件的读写

## common-utils

导入util下的其他头文件

## const-integer-set

整数集合，用于测试某个值是否在提供的集合当中

## edit-distance

计算编辑距离，实现了3种编辑距离的计算,分别为传统方法计算编辑距离，计算两个字符串之间的编辑距离，计算两个输入之前的对齐。

Levenshtein Distance 是用来度量两个序列相似程度的指标。通俗地来讲，编辑距离指的是在两个单词<w_1,w_2>之间，由其中一个单词w_1转换为另一个单词w_2所需要的最少**单字符编辑操作**次数

在这里定义的单字符编辑操作有且仅有三种：

- 插入（Insertion）
- 删除（Deletion）
- 替换（Substitution）

## hash-list

list的扩展，节点中包含key-value对和指向下一个节点的指针，主要用于解码获取当前帧数据和下一帧数据

## kaldi-holder

kaldi-holder类存储kaldi中的数据对象，可以通过read/write函数读入/写入磁盘文件，其中读写方式又可以分为binary和text

对于文本类型，每个元素要以换行符结尾

对于二进制类型，写入的元素会带有文件头，文件头中可以包含特征的帧数，维度，声学特征类型，占用字节数，是否压缩

util/kaldi-holder.h

```cpp
//holder的接口类，定义了holder使用的方法
template<class SomeType> class GenericHolder {
 public:
  typedef SomeType T;
  GenericHolder() { }
  static bool Write(std::ostream &os, bool binary, const T &t);
  bool Read(std::istream &is);
  static bool IsReadInBinary() { return true; }
  T &Value() { return t_; }  // if t is a pointer, would return *t_;
  void Clear() { }
  void Swap(GenericHolder<T> *other) { std::swap(t_, other->t_); }
  // ExtractRange提取特征的偏移量
  //例如mfcc.ark[:,0:20],表示mfcc特征的所有行和0~20列数据
  bool ExtractRange(const GenericHolder<T> &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }
  ~GenericHolder() { }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(GenericHolder); //限制不能使用=进行拷贝赋值
  T t_;  // t_ may alternatively be of type T*.
};
```

在util/kaldi-holder.h中还申明了kaldi中用到的所有holder

```cpp
/// KaldiObjectHolder仅对具有复制构造函数，默认构造函数和Kaldi Write和Read函数的Kaldi对象有效
// 如：Matrix 和 Vector
template<class KaldiType> class KaldiObjectHolder;

/// BasicHolder对float，double，bool和integer类型有效，否则会出现编译时错误
// 因为BasicHolder中使用的WriteBasicType/ReadBasicType只支持BasicType
template<class BasicType> class BasicHolder;


// 基本类型向量的Holder，例如 std :: vector <int32>，std :: vector <float>，依此类推。 
// 注意：基本类型定义为实现ReadBasicType和WriteBasicType的类型，即整数和浮点类型以及bool
template<class BasicType> class BasicVectorHolder;


// BasicVectorVectorHolder是基本类型的向量的向量的Holder，例如std::vector<std::vector<int32>>。 // 注意：基本类型定义为实现ReadBasicType和WriteBasicType的类型，即整数和浮点类型以及bool。
template<class BasicType> class BasicVectorVectorHolder;

//BasicPairVectorHolder是一个基本类型对的向量的Holder,如std::vector<std::pair<int32,int32>>。 // 注意：基本类型定义为实现ReadBasicType和WriteBasicType的类型，即整数和浮点类型以及bool。 
// Text format is (e.g. for integers), "1 12 ; 43 61 ; 17 8 \n"
template<class BasicType> class BasicPairVectorHolder;

/// 令牌定义为非空，可打印，无空格的string。 
// 令牌的二进制和文本格式是相同的（以换行符结尾），因此不必理会二进制模式的标头。
class TokenHolder;

/// TokenVectorHolder类是用于令牌向量的Holder类
/// (T == std::string).
class TokenVectorHolder;

/// HTK格式的读写，主要用于兼容HTK的数据格式，用到的不多
/// T == std::pair<Matrix<BaseFloat>, HtkHeader>
class HtkMatrixHolder;

/// Sphinx格式的读写，用到的不多
template<int kFeatDim = 13> class SphinxMatrixHolder;
```

在util/kaldi-holder.h中还申明了两种提取range的方法

```cpp
//从matrix或vector中截取数据赋值给另一个matrix或vector
bool ExtractObjectRange(const Matrix<Real> &input, const std::string &range,
                        Matrix<Real> *output);
//从命令行中提取range，如mfcc.ark[:,0:20]
bool ExtractRangeSpecifier(const std::string &rxfilename_with_range,
                           std::string *data_rxfilename,
                           std::string *range);
```

### KaldiObjectHolder

```cpp
//在kaldi源码中，kaldiType包含SparseMatrix<BaseFloat>，MatrixBase<BaseFloat>,
//Matrix<BaseFloat>,CompressedMatrix,VectorBase<BaseFloat>,Vector<BaseFloat>,
//CuMatrix<BaseFloat>,CuVector<BaseFloat>,GeneralMatrix,Supervision，AmDiagGmm，
//NnetExample,DiscriminativeNnetExample,NnetChainExample,NnetDiscriminativeExample,
//RnnlmExample,Sgmm2GauPost,NnetChainExample,RegtreeMllrDiagGmm,RegtreeFmllrDiagGmm
//以上类型在后面遇到再介绍
template<class KaldiType> class KaldiObjectHolder {
 public:
  typedef KaldiType T;

  KaldiObjectHolder(): t_(NULL) { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // 设置写入流是二进制还是文本格式，即是否写入\0B
    try {
      t.Write(os, binary);   //调用KaldiType对象的Write
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object. " << e.what();
      return false;  // Write failure.
    }
  }

  void Clear() {
    if (t_) {
      delete t_;
      t_ = NULL;
    }
  }

  bool Read(std::istream &is) {
    delete t_;
    t_ = new T;
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {  //判断流是否是二进制
      KALDI_WARN << "Reading Table object, failed reading binary header\n";
      return false;
    }
    try {
      t_->Read(is, is_binary);  //调用KaldiType对象的Read函数
      return true;
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object. " << e.what();
      delete t_;
      t_ = NULL;
      return false;
    }
  }

  // 无效的函数，在Read的时候会从流中判断是否是二进制
  static bool IsReadInBinary() { return true; }

  T &Value() {
    // code error if !t_.
    if (!t_) KALDI_ERR << "KaldiObjectHolder::Value() called wrongly.";
    return *t_;
  }

  void Swap(KaldiObjectHolder<T> *other) {
    // the t_ values are pointers so this is a shallow swap.
    std::swap(t_, other->t_);
  }

  bool ExtractRange(const KaldiObjectHolder<T> &other,
                    const std::string &range) {
    KALDI_ASSERT(other.t_ != NULL);
    delete t_;
    t_ = new T;
    //从matrix或vector中截取数据赋值给另一个matrix或vector
    return ExtractObjectRange(*(other.t_), range, t_);
  }

  ~KaldiObjectHolder() { delete t_; }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiObjectHolder);  //不允许使用=
  T *t_;
};
```

### BaseHolder

```cpp
//BasicType 有float, double, bool, and integer
template<class BasicType> class BasicHolder {
 public:
  typedef BasicType T;

  BasicHolder(): t_(static_cast<T>(-1)) { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // 设置写入流是二进制还是文本格式，即是否写入\0B
    try {
      WriteBasicType(os, binary, t);  //kaldi-base中实现的基础类型write
      if (!binary) os << '\n';  // 添加换行
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object. " << e.what();
      return false;  // Write failure.
    }
  }

  void Clear() { }

  // Reads into the holder.
  bool Read(std::istream &is) {
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary"
          " header\n";
      return false;
    }
    try {
      int c;
      if (!is_binary) {  
        // Eat up any whitespace and make sure it's not newline.
        while (isspace((c = is.peek())) && c != static_cast<int>('\n')) {
          is.get();
        }
        if (is.peek() == '\n') {
          KALDI_WARN << "Found newline but expected basic type.";
          return false;  // This is just to catch a more-
          // likely-than average type of error (empty line before the token),
          // since ReadBasicType will eat it up.
        }
      }

      ReadBasicType(is, is_binary, &t_);

      if (!is_binary) { 
        // make sure there is a newline.
        while (isspace((c = is.peek())) && c != static_cast<int>('\n')) {
          is.get();
        }
        if (is.peek() != '\n') {
          KALDI_WARN << "BasicHolder::Read, expected newline, got "
                     << CharToString(is.peek()) << ", position " << is.tellg();
          return false;
        }
        is.get();  // Consume the newline.
      }
      return true;
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object. " << e.what();
      return false;
    }
  }

  // 无效的函数，在Read的时候会从流中判断是否是二进制
  static bool IsReadInBinary() { return true; }

  T &Value() {
    return t_;
  }

  void Swap(BasicHolder<T> *other) {
    std::swap(t_, other->t_);
  }

  bool ExtractRange(const BasicHolder<T> &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  ~BasicHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicHolder);

  T t_;
};
```

### BasicVectorHolder和BasicVectorVectorHolder和BasicPairVectorHolder

BasicVectorHolder为读取基础类型的链表数据，和BasicHolder类似，可以自己看一下代码

BasicVectorVectorHolder为读取基础类型的链表链表数据，如std::vector<std::vector<int32> >，和BasicHolder类似，可以自己看一下代码

BasicPairVectorHolder是链表中包含字典的类型，如：std::vector<std::pair<int32, int32> >，可以自己看一下代码

### TokenHolder

TokenHolder数据为**非空，可打印，无空格**的string类型，不区分二进制和文本

```cpp
class TokenHolder {
 public:
  typedef std::string T;

  TokenHolder() {}

  static bool Write(std::ostream &os, bool, const T &t) {  // ignore binary-mode
    KALDI_ASSERT(IsToken(t));
    os << t << '\n';
    return os.good();
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    is >> t_;
    if (is.fail()) return false;
    char c;
    while (isspace(c = is.peek()) && c!= '\n') is.get();
    if (is.peek() != '\n') {
      KALDI_WARN << "TokenHolder::Read, expected newline, got char "
        << CharToString(is.peek())
        << ", at stream pos " << is.tellg();
      return false;
    }
    is.get();  // get '\n'
    return true;
  }


  // Since this is fundamentally a text format, read in text mode (would work
  // fine either way, but doing it this way will exercise more of the code).
  static bool IsReadInBinary() { return false; }

  T &Value() { return t_; }

  ~TokenHolder() { }

  void Swap(TokenHolder *other) {
    t_.swap(other->t_);
  }

  bool ExtractRange(const TokenHolder &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(TokenHolder);
  T t_;
};
```

### TokenVectorHolder

token的链表类型，可以自己看看源码

### HtkMatrixHolder和SphinxMatrixHolder

读取htk和Sphinx格式的数据，现在基本用不到

## kaldi-io

kaldi中非常重要的部分，实现了文件/标准输入输出/管道的读写。

kaldi-io.h中定义了输入和输出的类型，输入输出的入口类

```cpp
enum OutputType {
  kNoOutput,      //无效的输出
  kFileOutput,	//文件输出
  kStandardOutput,  //标准输出，一般为输出到控制台，使用""或"-"
  kPipeOutput  //管道输出，例如"| gzip -c > /tmp/abc.gz"
};
//判断命令行输出的形式
OutputType ClassifyWxfilename(const std::string &wxfilename);

enum InputType {
  kNoInput,  //无效的输入
  kFileInput,  //文件输入
  kStandardInput,  //标准输入，使用""或"-"
  kOffsetFileInput,  //文件偏移输入，如/some/filename:12970；在kaldi中会将多个数据写入一个文件，使用文件偏移输入可以快速读取相应的数据
  kPipeInput  //管道输入，如："gunzip -c /tmp/abc.gz |"
};
//判断命令行输入的形式
InputType ClassifyRxfilename(const std::string &rxfilename);
```

```cpp
class Output {
 public:
  Output(const std::string &filename, bool binary, bool write_header = true);
  Output(): impl_(NULL) {}
  bool Open(const std::string &wxfilename, bool binary, bool write_header);
  inline bool IsOpen(); 
  std::ostream &Stream();  
  bool Close();
  ~Output();
 private:
  OutputImplBase *impl_;  //实际输出类型的代理，如StandardOutputImpl，PipeOutputImpl，FileOutputImpl
  std::string filename_;  //输入的命令行
  KALDI_DISALLOW_COPY_AND_ASSIGN(Output);
};
```

```cpp
class Input {
 public:
  Input(const std::string &rxfilename, bool *contents_binary = NULL);

  Input(): impl_(NULL) {}
  inline bool Open(const std::string &rxfilename, bool *contents_binary = NULL);
  inline bool OpenTextMode(const std::string &rxfilename);
  inline bool IsOpen();
  int32 Close();
  std::istream &Stream();
  ~Input();
 private:
  bool OpenInternal(const std::string &rxfilename, bool file_binary,
                    bool *contents_binary);  //Input内部使用，判断
  InputImplBase *impl_;  //实际输入的代理，如FileInputImpl，StandardInputImpl，PipeInputImpl，OffsetFileInputImpl
  KALDI_DISALLOW_COPY_AND_ASSIGN(Input);
};
```

除了通过Input读取命令行数据，Output写入，kaldi还支持将文件直接读取/写入文件；

和Input、Output的区别在于：Input、Output输入的格式为ark:path/input.ark:1234，直接读取的时候输入的格式为path/input.ark:1234

```cpp
template <class C> void ReadKaldiObject(const std::string &filename,
                                        C *c) {
  bool binary_in;
  Input ki(filename, &binary_in);
  c->Read(ki.Stream(), binary_in);
}
template <> void ReadKaldiObject(const std::string &filename,
                                 Matrix<float> *m);
template <> void ReadKaldiObject(const std::string &filename,
                                 Matrix<double> *m);
template <class C> inline void WriteKaldiObject(const C &c,const std::string &filename,
                                                bool binary) {
  Output ko(filename, binary);
  c.Write(ko.Stream(), binary);
}
```

```cpp
//将rxfilename，wxfilename转换为更易于理解的形式
std::string PrintableRxfilename(const std::string &rxfilename);
std::string PrintableWxfilename(const std::string &wxfilename);
```

kaldi-io-inl.h实现了kaldi-io.h中的open和isopen方法

```cpp
bool Input::Open(const std::string &rxfilename, bool *binary) {
  return OpenInternal(rxfilename, true, binary);
}

bool Input::OpenTextMode(const std::string &rxfilename) {
  return OpenInternal(rxfilename, false, NULL);
}

bool Input::IsOpen() {
  return impl_ != NULL;
}

bool Output::IsOpen() {
  return impl_ != NULL;
}
```

## kaldi-semaphore

kaldi自己实现的信号量机制，用于线程间同步

```cpp
class Semaphore {
 public:
  Semaphore(int32 count = 0);

  ~Semaphore();

  bool TryWait();  ///申请锁，如果成功返回true，失败返回false
  void Wait();     ///申请锁，成功返回void，失败会阻塞，直到获取锁
  void Signal();   ///释放锁

 private:
  int32 count_;    ///< the semaphore counter, 0 means block on Wait()

  std::mutex mutex_;
  std::condition_variable condition_variable_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Semaphore);
};
```

## kaldi-table

table可以理解为string,holder组成的vector或者map，是一组数据的集合

```cpp
enum WspecifierType  {
  kNoWspecifier,       //错误的类型
  kArchiveWspecifier,  //ark:filename
  kScriptWspecifier,  //scp:wxfilename
  kBothWspecifier  //ark,scp:filename,wxfilename，ark和scp不能互换位置
};
//写表单属性
struct WspecifierOptions {
  bool binary;  //b:二进制模式,t:文本模式
  bool flush;  //f；nf。用于确定在每次写操作后是否刷新数据流，默认是f，即刷新，这样有利于优化内存使用
  bool permissive;  //宽容模式：p。只对输出列表表单(scp)有效。例如，在同时输出存档表单和列表表单时，
    //如果表单的某个元素对应的存档内容无法获取，那么在列表表单中将直接跳过这个元素，不提示错误
  WspecifierOptions(): binary(true), flush(false), permissive(false) { }
};
//解析命令行
WspecifierType ClassifyWspecifier(const std::string &wspecifier,
                                  std::string *archive_wxfilename,
                                  std::string *script_wxfilename,
                                  WspecifierOptions *opts);

enum RspecifierType  {
  kNoRspecifier,       //错误的类型
  kArchiveRspecifier,  //ark:rxfilename
  kScriptRspecifier    //scp:rxfilename
};
struct  RspecifierOptions {
  // These options only make a difference for the RandomAccessTableReader class.
  bool once; //o；no。告知可执行程序，在读入表单中每个索引只出现一次，不会出现多个元素使用一个索引的情况
  bool sorted;  //s；ns。在输入的表单中，元素的索引是有序的，这个有序是字符串意义上的
  bool called_sorted;  // cs；ncs。表单中的元素将被顺序访问
  bool permissive;  // p；np。如果输入的列表表单中某个元素的目标文件无法获取或输入的存档表单中
                    //某个元素的内容有误，则不会抛出错误，而是在日志中打印一个警告
  bool background;  // bg,在后台线程中“提前读取”下一个值。在读取较大的对象（例如神经网络训练示例）时，
    //尤其在要最大程度地利用GPU的情况下，建议使用此选项。
  RspecifierOptions(): once(false), sorted(false),
                       called_sorted(false), permissive(false),
                       background(false) { }
};
// 解析命令行
RspecifierType ClassifyRspecifier(const std::string &rspecifier,
                                  std::string *rxfilename,
                                  RspecifierOptions *opts);
```

### RandomAccessTableReader

索引方式读取数据

```cpp
template<class Holder>
class RandomAccessTableReader {
 public:
  typedef typename Holder::T T;
  RandomAccessTableReader(): impl_(NULL) { }
  // 在该构造函数中会调用open
  explicit RandomAccessTableReader(const std::string &rspecifier);
  // Opens the table.
  bool Open(const std::string &rspecifier);
  // Returns true if table is open.
  bool IsOpen() const { return (impl_ != NULL); }
  bool Close();
  // 在 "permissive" (p) 模式下,如果不存在key，返回false
  bool HasKey(const std::string &key);
  const T &Value(const std::string &key);
  ~RandomAccessTableReader();

  // Allow copy-constructor only for non-opened readers (needed for inclusion in
  // stl vector)
  RandomAccessTableReader(const RandomAccessTableReader<Holder>
                          &other):
      impl_(NULL) { KALDI_ASSERT(other.impl_ == NULL); }
 private:
  // Disallow assignment.
  RandomAccessTableReader &operator=(const RandomAccessTableReader<Holder>&);
  void CheckImpl() const;  // Checks that impl_ is non-NULL; prints an error
                           // message and dies (with KALDI_ERR) if NULL.
  //代理，实际为RandomAccessTableReaderScriptImpl: scp根据索引读取
  //RandomAccessTableReaderArchiveImplBase  
  //RandomAccessTableReaderDSortedArchiveImpl:double sort,s和cs选项同时设置时才生效
  //RandomAccessTableReaderSortedArchiveImpl:s选项设置，cs选线未设置时使用
  //RandomAccessTableReaderUnsortedArchiveImpl:s和cs选项都未设置时使用
  RandomAccessTableReaderImplBase<Holder> *impl_;  
};
```

### SequentialTableReader

顺序表读取，kaldi中顺序表是根据字符串排序后的表。

```cpp
template<class Holder>
class SequentialTableReader {
 public:
  typedef typename Holder::T T;
  SequentialTableReader(): impl_(NULL) { }
  // 在该构造函数中会调用open
  explicit SequentialTableReader(const std::string &rspecifier);
  bool Open(const std::string &rspecifier);
  inline bool Done();
  // Only valid to call Key() if Done() returned false.
  inline std::string Key();
  // FreeCurrent()是为节省大型对象的内存而优化提供的。释放当前value的内存
  void FreeCurrent();
  T &Value();
  void Next();
  bool IsOpen() const;
  bool Close();
  ~SequentialTableReader();

  // Allow copy-constructor only for non-opened readers (needed for inclusion in
  // stl vector)
  SequentialTableReader(const SequentialTableReader<Holder> &other):
      impl_(NULL) { KALDI_ASSERT(other.impl_ == NULL); }
 private:
  // Disallow assignment.
  SequentialTableReader &operator = (const SequentialTableReader<Holder>&);
  void CheckImpl() const;  // Checks that impl_ is non-NULL; prints an error
                           // message and dies (with KALDI_ERR) if NULL.
  //代理，实际为：SequentialTableReaderScriptImpl:读scp文件
  //SequentialTableReaderArchiveImpl:读ark文件
  //SequentialTableReaderBackgroundImpl:在后台线程中“提前读取”下一个值
  SequentialTableReaderImplBase<Holder> *impl_;
};
```

### TableWriter

将kaldi对象写入到文件

```cpp
template<class Holder>
class TableWriter {
 public:
  typedef typename Holder::T T;

  TableWriter(): impl_(NULL) { }

  // 在该构造函数中会调用open
  explicit TableWriter(const std::string &wspecifier);
  bool Open(const std::string &wspecifier);
  bool IsOpen() const;
  inline void Write(const std::string &key, const T &value) const;
  void Flush();
  bool Close();
  ~TableWriter();
  // Allow copy-constructor only for non-opened writers (needed for inclusion in
  // stl vector)
  TableWriter(const TableWriter &other): impl_(NULL) {
    KALDI_ASSERT(other.impl_ == NULL);
  }
 private:
  TableWriter &operator = (const TableWriter&);  // Disallow assignment.

  void CheckImpl() const;  // Checks that impl_ is non-NULL; prints an error
                           // message and dies (with KALDI_ERR) if NULL.
  //代理，实际为：TableWriterArchiveImpl:写ark文件
  //TableWriterScriptImpl:写scp文件
  //TableWriterBothImpl:写ark,scp文件
  TableWriterImplBase<Holder> *impl_;
};
```

### RandomAccessTableReaderMapped

随机访问，一般用于utt2spk文件解析

utt1 spk1
utt2 spk1
utt3 spk1

```cpp
template<class Holder>
class RandomAccessTableReaderMapped {
 public:
  typedef typename Holder::T T;
  /// Note: "utt2spk_rxfilename" will in the normal case be an rxfilename
  /// for an utterance to speaker map, but this code is general; it accepts
  /// a generic map.
  RandomAccessTableReaderMapped(const std::string &table_rxfilename,
                                const std::string &utt2spk_rxfilename);
  RandomAccessTableReaderMapped() {}

  /// Note: when calling Open, utt2spk_rxfilename may be empty.
  bool Open(const std::string &table_rxfilename,
            const std::string &utt2spk_rxfilename);

  bool HasKey(const std::string &key);
  const T &Value(const std::string &key);
  inline bool IsOpen() const { return reader_.IsOpen(); }
  inline bool Close() { return reader_.Close(); }
  // The default copy-constructor will do what we want: it will crash for
  // already-opened readers, by calling the member-variable copy-constructors.
 private:
  // Disallow assignment.
  RandomAccessTableReaderMapped &operator =
    (const RandomAccessTableReaderMapped<Holder>&);
  RandomAccessTableReader<Holder> reader_;
  RandomAccessTableReader<TokenHolder> token_reader_;
  std::string utt2spk_rxfilename_;  // Used only in diagnostic messages.
};
```

## kaldi-thread

kaldi-thread实现了线程池，任务队列

在kaldi的部分bin中支持多线程，通过--num-threads命令设置线程数，最终会赋值给kaldi-thread中的g_num_threads用来控制线程池大小。

## parse-options/simple-options

kaldi命令行解析实现，感兴趣可以详细看看。

## simple-io-funcs

只是实现了对int类型的读写

```cpp
bool WriteIntegerVectorSimple(const std::string &wxfilename,
                              const std::vector<int32> &v);
bool ReadIntegerVectorSimple(const std::string &rxfilename,
                             std::vector<int32> *v);
bool WriteIntegerVectorVectorSimple(const std::string &wxfilename,
                                    const std::vector<std::vector<int32> > &v);
bool ReadIntegerVectorVectorSimple(const std::string &rxfilename,
                                   std::vector<std::vector<int32> > *v);
```

## stl-utils

对数据的一些基本操作如：

```cpp
inline void SortAndUniq(std::vector<T> *vec);
inline bool IsSorted(const std::vector<T> &vec);
inline bool IsSortedAndUniq(const std::vector<T> &vec);
inline void Uniq(std::vector<T> *vec);
void CopySetToVector(const std::set<T> &s, std::vector<T> *v);
void CopySetToVector(const unordered_set<T> &s, std::vector<T> *v);
void CopyMapToVector(const std::map<A, B> &m,std::vector<std::pair<A, B> > *v);
void CopyMapKeysToVector(const std::map<A, B> &m, std::vector<A> *v);
void CopyMapValuesToVector(const std::map<A, B> &m, std::vector<B> *v);
void CopyMapKeysToSet(const std::map<A, B> &m, std::set<A> *s);
void CopyMapValuesToSet(const std::map<A, B> &m, std::set<B> *s);
void CopyVectorToSet(const std::vector<A> &v, std::set<A> *s);
void DeletePointers(std::vector<A*> *v);
bool ContainsNullPointers(const std::vector<A*> &v);
void CopyVectorToVector(const std::vector<A> &vec_in, std::vector<B> *vec_out);
inline void ReverseVector(std::vector<T> *vec);
inline void MergePairVectorSumming(std::vector<std::pair<I, F> > *vec);
```

## table-types

table-types定义了kaldi中所有使用到的table类型

```cpp
typedef TableWriter<KaldiObjectHolder<MatrixBase<BaseFloat> > >
                    BaseFloatMatrixWriter;
typedef SequentialTableReader<KaldiObjectHolder<Matrix<BaseFloat> > >
                             SequentialBaseFloatMatrixReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Matrix<BaseFloat> > >
                                RandomAccessBaseFloatMatrixReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<Matrix<BaseFloat> > >
                                      RandomAccessBaseFloatMatrixReaderMapped;
typedef TableWriter<KaldiObjectHolder<MatrixBase<double> > >
                                      DoubleMatrixWriter;
typedef SequentialTableReader<KaldiObjectHolder<Matrix<double> > >
                              SequentialDoubleMatrixReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Matrix<double> > >
                                RandomAccessDoubleMatrixReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<Matrix<double> > >
                                      RandomAccessDoubleMatrixReaderMapped;
typedef TableWriter<KaldiObjectHolder<CompressedMatrix> >
                                      CompressedMatrixWriter;
typedef TableWriter<KaldiObjectHolder<VectorBase<BaseFloat> > >
                                      BaseFloatVectorWriter;
typedef SequentialTableReader<KaldiObjectHolder<Vector<BaseFloat> > >
                              SequentialBaseFloatVectorReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Vector<BaseFloat> > >
                                RandomAccessBaseFloatVectorReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<Vector<BaseFloat> > >
                                      RandomAccessBaseFloatVectorReaderMapped;
typedef TableWriter<KaldiObjectHolder<VectorBase<double> > >
                                      DoubleVectorWriter;
typedef SequentialTableReader<KaldiObjectHolder<Vector<double> > >
                              SequentialDoubleVectorReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Vector<double> > >
                                RandomAccessDoubleVectorReader;
typedef TableWriter<KaldiObjectHolder<CuMatrix<BaseFloat> > >
                                      BaseFloatCuMatrixWriter;
typedef SequentialTableReader<KaldiObjectHolder<CuMatrix<BaseFloat> > >
                              SequentialBaseFloatCuMatrixReader;
typedef RandomAccessTableReader<KaldiObjectHolder<CuMatrix<BaseFloat> > >
                                RandomAccessBaseFloatCuMatrixReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<CuMatrix<BaseFloat> > >
                                      RandomAccessBaseFloatCuMatrixReaderMapped;
typedef TableWriter<KaldiObjectHolder<CuMatrix<double> > >
                                      DoubleCuMatrixWriter;
typedef SequentialTableReader<KaldiObjectHolder<CuMatrix<double> > >
                              SequentialDoubleCuMatrixReader;
typedef RandomAccessTableReader<KaldiObjectHolder<CuMatrix<double> > >
                                RandomAccessDoubleCuMatrixReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<CuMatrix<double> > >
                                      RandomAccessDoubleCuMatrixReaderMapped;
typedef TableWriter<KaldiObjectHolder<CuVector<BaseFloat> > >
                    BaseFloatCuVectorWriter;
typedef SequentialTableReader<KaldiObjectHolder<CuVector<BaseFloat> > >
                              SequentialBaseFloatCuVectorReader;
typedef RandomAccessTableReader<KaldiObjectHolder<CuVector<BaseFloat> > >
                                RandomAccessBaseFloatCuVectorReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<CuVector<BaseFloat> > >
                                      RandomAccessBaseFloatCuVectorReaderMapped;
typedef TableWriter<KaldiObjectHolder<CuVector<double> > >
                    DoubleCuVectorWriter;
typedef SequentialTableReader<KaldiObjectHolder<CuVector<double> > >
                              SequentialDoubleCuVectorReader;
typedef RandomAccessTableReader<KaldiObjectHolder<CuVector<double> > >
                                RandomAccessDoubleCuVectorReader;
typedef TableWriter<BasicHolder<int32> >  Int32Writer;
typedef SequentialTableReader<BasicHolder<int32> >  SequentialInt32Reader;
typedef RandomAccessTableReader<BasicHolder<int32> >  RandomAccessInt32Reader;
typedef TableWriter<BasicVectorHolder<int32> >  Int32VectorWriter;
typedef SequentialTableReader<BasicVectorHolder<int32> >
                              SequentialInt32VectorReader;
typedef RandomAccessTableReader<BasicVectorHolder<int32> >
                                RandomAccessInt32VectorReader;
typedef TableWriter<BasicVectorVectorHolder<int32> >  Int32VectorVectorWriter;
typedef SequentialTableReader<BasicVectorVectorHolder<int32> >
                              SequentialInt32VectorVectorReader;
typedef RandomAccessTableReader<BasicVectorVectorHolder<int32> >
                                RandomAccessInt32VectorVectorReader;
typedef TableWriter<BasicPairVectorHolder<int32> >  Int32PairVectorWriter;
typedef SequentialTableReader<BasicPairVectorHolder<int32> >
                              SequentialInt32PairVectorReader;
typedef RandomAccessTableReader<BasicPairVectorHolder<int32> >
                                RandomAccessInt32PairVectorReader;
typedef TableWriter<BasicPairVectorHolder<BaseFloat> >
                    BaseFloatPairVectorWriter;
typedef SequentialTableReader<BasicPairVectorHolder<BaseFloat> >
                              SequentialBaseFloatPairVectorReader;
typedef RandomAccessTableReader<BasicPairVectorHolder<BaseFloat> >
                                RandomAccessBaseFloatPairVectorReader;
typedef TableWriter<BasicHolder<BaseFloat> >  BaseFloatWriter;
typedef SequentialTableReader<BasicHolder<BaseFloat> >
                              SequentialBaseFloatReader;
typedef RandomAccessTableReader<BasicHolder<BaseFloat> >
                                RandomAccessBaseFloatReader;
typedef RandomAccessTableReaderMapped<BasicHolder<BaseFloat> >
                                      RandomAccessBaseFloatReaderMapped;
typedef TableWriter<BasicHolder<double> >  DoubleWriter;
typedef SequentialTableReader<BasicHolder<double> >  SequentialDoubleReader;
typedef RandomAccessTableReader<BasicHolder<double> >  RandomAccessDoubleReader;

typedef TableWriter<BasicHolder<bool> >  BoolWriter;
typedef SequentialTableReader<BasicHolder<bool> >  SequentialBoolReader;
typedef RandomAccessTableReader<BasicHolder<bool> >  RandomAccessBoolReader;
/// TokenWriter is a writer specialized for std::string where the strings
/// are nonempty and whitespace-free.   T == std::string
typedef TableWriter<TokenHolder> TokenWriter;
typedef SequentialTableReader<TokenHolder> SequentialTokenReader;
typedef RandomAccessTableReader<TokenHolder> RandomAccessTokenReader;
/// TokenVectorWriter is a writer specialized for sequences of
/// std::string where the strings are nonempty and whitespace-free.
/// T == std::vector<std::string>
typedef TableWriter<TokenVectorHolder> TokenVectorWriter;
// Ditto for SequentialTokenVectorReader.
typedef SequentialTableReader<TokenVectorHolder> SequentialTokenVectorReader;
typedef RandomAccessTableReader<TokenVectorHolder>
                                RandomAccessTokenVectorReader;
typedef TableWriter<KaldiObjectHolder<GeneralMatrix> >
                                      GeneralMatrixWriter;
typedef SequentialTableReader<KaldiObjectHolder<GeneralMatrix> >
                              SequentialGeneralMatrixReader;
typedef RandomAccessTableReader<KaldiObjectHolder<GeneralMatrix> >
                                RandomAccessGeneralMatrixReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<GeneralMatrix> >
                                      RandomAccessGeneralMatrixReaderMapped;
```

## text-utils

kaldi中对字符串处理的方法，如：

```cpp
//字符串分割，如果omit_empty_strings为true，则对分割后的字符串去空格
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out);
//字符串连接，如果omit_empty_strings为true，则跳过所有空字符串
void JoinVectorToString(const std::vector<std::string> &vec_in,
                        const char *delim, bool omit_empty_strings,
                        std::string *str_out);
//将split之后的字符串转换为long long integer
template<class I>
bool SplitStringToIntegers(const std::string &full,
                           const char *delim,
                           bool omit_empty_strings,  // typically false [but
                                                     // should probably be true
                                                     // if "delim" is spaces].
                           std::vector<I> *out);
//将split之后的字符串转化为浮点类型
template<class F>
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out);
template<class Int>
bool ConvertStringToInteger(const std::string &str,Int *out);
template <typename T>
bool ConvertStringToReal(const std::string &str,T *out);
//从字符串中删除开头和结尾的空格
void Trim(std::string *str);
//删除字符串首尾空格，在使用空格将字符串分为两部分，第一部分放到first,第二部分放到rest，如果不存在分割的空格，则将字符串放到first
void SplitStringOnFirstSpace(const std::string &line,
                             std::string *first,
                             std::string *rest);
//非空且所有字符均可打印且无空格，则返回true。
bool IsToken(const std::string &token);
//没有\ n字符和不可打印的字符，并且不包含前导或尾随空格，则返回true。
bool IsLine(const std::string &line);
//判断字符串是否近似相等，decimal_places_check控制'.'之后的位数
//例如：StringsApproxEqual("hello 0.23 there", "hello 0.24 there", 2)返回false，如果
//decimal_places_check为1，则0.23和0.24近似相等，返回true
bool StringsApproxEqual(const std::string &a,
                        const std::string &b,
                        int32 decimal_places_check = 2);
//接受token1+token2或仅接受token2
void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                          const std::string &token1,
                          const std::string &token2);
//读配置文件，把配置文件中每行配置保存到lines中
void ReadConfigLines(std::istream &is,
                     std::vector<std::string> *lines);
void ParseConfigLines(const std::vector<std::string> &lines,
                      std::vector<ConfigLine> *config_lines);
//以A-Za-z_开头的非空字符串，仅包含'-'，'_'，'.'，A-Z，a-z或0-9。
bool IsValidName(const std::string &name);
```



