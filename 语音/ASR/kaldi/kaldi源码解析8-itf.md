# kaldi源码解析8-itf
itf是interface的缩写，在itf目录下定义各种扩展接口

## clusterable-itf

聚类接口

```cpp
class Clusterable {
 public:
  /// Return a copy of this object.
  virtual Clusterable *Copy() const = 0;
  /// 返回与统计信息相关联的目标函数[假设ML估算]
  virtual BaseFloat Objf() const = 0;
  /// 返回与stats相关的归一化器（通常为计数）
  virtual BaseFloat Normalizer() const = 0;
  /// Set stats to empty.
  virtual void SetZero() = 0;
  /// Add other stats.
  virtual void Add(const Clusterable &other) = 0;
  /// Subtract other stats.
  virtual void Sub(const Clusterable &other) = 0;
  /// Scale the stats by a positive number f [not mandatory to supply this].
  virtual void Scale(BaseFloat f) {
    KALDI_ERR << "This Clusterable object does not implement Scale().";
  }

  /// Return a string that describes the inherited type. 
  virtual std::string Type() const = 0;

  /// Write data to stream.
  virtual void Write(std::ostream &os, bool binary) const = 0;

  /// 从流中读取数据并返回相应的对象
  virtual Clusterable* ReadNew(std::istream &os, bool binary) const = 0;
  virtual ~Clusterable() {}
  /// 两个类别相加
  virtual BaseFloat ObjfPlus(const Clusterable &other) const;
  /// 两个类别相减
  virtual BaseFloat ObjfMinus(const Clusterable &other) const;
  /// 计算两个类别的距离，大于等于0
  virtual BaseFloat Distance(const Clusterable &other) const;
};
```

## context-dep-itf

context-dep-itf.h提供tree中的树构建代码与fstext中的FST代码之间的链接。 它是一个抽象接口，描述了一个对象，该对象可以从上下文音素映射到整数叶ID序列。

```cpp
class ContextDependencyInterface {
 public:
  /// ContextWidth()返回值N（例如，三音素为3），该值表示计算上下文时要考虑的音素数量。
  virtual int ContextWidth() const = 0;

  /// 音素上下文的中心位置P，从0开始的编号，三音系统P=1
  virtual int CentralPosition() const = 0;

  /// 用于经典的拓扑，pdf_class为0、1、2。返回成功或失败；否则，返回0。 输出pdf-id。
  virtual bool Compute(const std::vector<int32> &phoneseq, int32 pdf_class,
                       int32 *pdf_id) const = 0;

  /// GetPdfInfo返回一个以pdf-id索引的向量，表示每个pdf可以对应哪些(phone, pdf-class)对
  virtual void GetPdfInfo(
      const std::vector<int32> &phones,  // list of phones
      const std::vector<int32> &num_pdf_classes,  // indexed by phone,
      std::vector<std::vector<std::pair<int32, int32> > > *pdf_info)
      const = 0;

  // 输出有关可以为HMM状态生成什么样的pdf-id的信息,比其他版本的GetPdfInfo（）效率低
  virtual void GetPdfInfo(
      const std::vector<int32> &phones,
      const std::vector<std::vector<std::pair<int32, int32> > > &pdf_class_pairs,
      std::vector<std::vector<std::vector<std::pair<int32, int32> > > > *pdf_info)
      const = 0;

  /// 返回声学pdf的数量
  virtual int32 NumPdfs() const = 0;
  virtual ~ContextDependencyInterface() {};
  ContextDependencyInterface() {}

  /// Returns pointer to new object which is copy of current one.
  virtual ContextDependencyInterface *Copy() const = 0;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(ContextDependencyInterface);
};
```

## decodable-itf

解码接口

```cpp
class DecodableInterface {
 public:
  /// 返回某一帧frame的某一个状态index的对数似然，它将在解码器中取反。 “帧”从零开始。 调用此方法之前，应验证NumFramesReady()> frame。
  virtual BaseFloat LogLikelihood(int32 frame, int32 index) = 0;

  /// 如果这是最后一帧，则返回true  online解码使用
  virtual bool IsLastFrame(int32 frame) const = 0;

  /// 返回当前可解码对象可用的帧数,用于不希望解码器在等待输入时阻塞的设置中 online2解码使用
  virtual int32 NumFramesReady() const {
    KALDI_ERR << "NumFramesReady() not implemented for this decodable type.";
    return -1;
  }

  /// 返回声学模型中的状态数（它们将从1开始索引，即从1到NumIndices（）；这是为了与OpenFst兼容）
  virtual int32 NumIndices() const = 0;

  virtual ~DecodableInterface() {}
};
```

## online-feature-itf

在线特征提取接口，在online2 /目录中使用的，它取代了../online/online-feat-input.h中的接口

```cpp
class OnlineFeatureInterface {
 public:
  virtual int32 Dim() const = 0; /// returns the feature dimension.

  /// 返回从说话开始到现在的总帧数，在在线解码的场景下，随着可用数据增加会增加
  virtual int32 NumFramesReady() const = 0;

  /// 如果这是最后一帧，则返回true
  virtual bool IsLastFrame(int32 frame) const = 0;

  // 获取此帧的特征向量。 在为给定帧调用此函数之前，假定您已调用NumFramesReady（），
  // 并且返回的数字大于“ frame”。 否则，此调用可能会因断言失败而崩溃
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) = 0;


  /// 获取一组帧向量，默认的实现方式是一帧一帧地获取，但子类可能会为了效率而对其进行覆盖（因为有时批量处理会更高效）。
  virtual void GetFrames(const std::vector<int32> &frames,
                         MatrixBase<BaseFloat> *feats) {
    KALDI_ASSERT(static_cast<int32>(frames.size()) == feats->NumRows());
    for (size_t i = 0; i < frames.size(); i++) {
      SubVector<BaseFloat> feat(*feats, i);
      GetFrame(frames[i], &feat);
    }
  }
  // 返回以秒为单位的帧偏移。 帮助估计帧数的持续时间。
  virtual BaseFloat FrameShiftInSeconds() const = 0;

  /// Virtual destructor.  Note: constructors that take another member of
  /// type OnlineFeatureInterface are not expected to take ownership of
  /// that pointer; the caller needs to keep track of that manually.
  virtual ~OnlineFeatureInterface() { }

};
```

```cpp
//mfcc fbank plp虚拟基类
class OnlineBaseFeature: public OnlineFeatureInterface {
 public:
  /// 当您获得更多wave数据时，将从应用程序中调用此方法
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform) = 0;

  /// InputFinished()不再提供任何波形
  virtual void InputFinished() = 0;
};
```

## optimizable-itf

优化器接口

```cpp
template<class Real>
class OptimizableInterface {
 public:
  /// 计算梯度
  virtual void ComputeGradient(const Vector<Real> &params,
                               Vector<Real> *gradient_out) = 0;
  /// 计算优化后的参数
  virtual Real ComputeValue(const Vector<Real> &params) = 0;

  virtual ~OptimizableInterface() {}
};
```

## options-itf

命令行解析接口

```cpp
class OptionsItf {
 public:
  virtual void Register(const std::string &name,
                bool *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                int32 *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                uint32 *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                float *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                double *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                std::string *ptr, const std::string &doc) = 0; 
  
  virtual ~OptionsItf() {}
};
```





