# kaldi源码解析9-feat

kaldi-feat是kaldi中提取音频特征的代码实现，包括mfcc，fbank和plp特征

## feature-common

```cpp
//离线特征提取，即一开始就可以访问整个语音
//F类型有FbankComputer，MfccComputer，PlpComputer，SpectrogramComputer
template <class F>
class OfflineFeatureTpl {
 public:
  typedef typename F::Options Options;
  OfflineFeatureTpl(const Options &opts):
      computer_(opts), feature_window_function_(computer_.GetFrameOptions()) { }
  // 内部接口，要求音频采样率等于option中配置的采样率
  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,Matrix<BaseFloat> *output);

  // const版本，会调用非const版本
  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,
               Matrix<BaseFloat> *output) const;

  /**计算特征，可以指定采样率
     @param [in] wave   The input waveform
     @param [in] sample_freq  采样率，如果大于config中采样率，则下采样，否则报错
     @param [in] vtln_warp  VTLN系数 (通常为1.0)
  */
  void ComputeFeatures(const VectorBase<BaseFloat> &wave,
                       BaseFloat sample_freq,
                       BaseFloat vtln_warp,
  int32 Dim() const { return computer_.Dim(); }
  // Copy constructor.
  OfflineFeatureTpl(const OfflineFeatureTpl<F> &other):
      computer_(other.computer_),
      feature_window_function_(other.feature_window_function_) { }
  private:
  // Disallow assignment.
  OfflineFeatureTpl<F> &operator =(const OfflineFeatureTpl<F> &other);
  F computer_;
  FeatureWindowFunction feature_window_function_;
};
```



## mel-computations

计算梅尔频谱

```cpp
struct MelBanksOptions {
  int32 num_bins;  // e.g. 25; 三角滤波器,16k语音建议为23,8k语音建议为15
  BaseFloat low_freq;  // e.g. 20; 低频截断
  BaseFloat high_freq;  // 高频截断
  BaseFloat vtln_low;  // vtln lower cutoff of warping function.
  BaseFloat vtln_high;  // vtln upper cutoff of warping function: if negative, added
                        // to the Nyquist frequency to get the cutoff.
  bool debug_mel;  //打印调试信息
  bool htk_mode;
  explicit MelBanksOptions(int num_bins = 25)
      : num_bins(num_bins), low_freq(20), high_freq(0), vtln_low(100),
        vtln_high(-500), debug_mel(false), htk_mode(false) {}
  void Register(OptionsItf *opts) {
    opts->Register("num-mel-bins", &num_bins,
                   "Number of triangular mel-frequency bins");
    opts->Register("low-freq", &low_freq,
                   "Low cutoff frequency for mel bins");
    opts->Register("high-freq", &high_freq,
                   "High cutoff frequency for mel bins (if <= 0, offset from Nyquist)");
    opts->Register("vtln-low", &vtln_low,
                   "Low inflection point in piecewise linear VTLN warping function");
    opts->Register("vtln-high", &vtln_high,
                   "High inflection point in piecewise linear VTLN warping function"
                   " (if negative, offset from high-mel-freq");
    opts->Register("debug-mel", &debug_mel,
                   "Print out debugging information for mel bin computation");
  }
};
```

## feature-window

窗口操作实现

```cpp
struct FrameExtractionOptions {
  BaseFloat samp_freq;  //采样率
  BaseFloat frame_shift_ms;  // 帧移，ms.
  BaseFloat frame_length_ms;  // 帧长，ms.
  BaseFloat dither;  // 抖动量，0.0表示无抖动.
  BaseFloat preemph_coeff;  // 预加重系数
  bool remove_dc_offset;  // 减去FFT前的均值.
  std::string window_type;//窗类型"hamming","rectangular",
                          //"povey","hanning","sine", "blackman"
  bool round_to_power_of_two;//如果为true，则通过FFT的零填充输入将窗口大小舍入为2的幂
  BaseFloat blackman_coeff;//广义布莱克曼窗的常数系数。
  bool snip_edges;  //是否进行边缘修剪
  bool allow_downsample;  //允许下采样
  bool allow_upsample;  //允许上采样
  int max_feature_vectors; //内存优化.如果大于0，则定期删除特征向量，以便仅保留此数量的最新特征向量。
  FrameExtractionOptions():
      samp_freq(16000),
      frame_shift_ms(10.0),
      frame_length_ms(25.0),
      dither(1.0),
      preemph_coeff(0.97),
      remove_dc_offset(true),
      window_type("povey"),
      round_to_power_of_two(true),
      blackman_coeff(0.42),
      snip_edges(true),
      allow_downsample(false),
      allow_upsample(false),
      max_feature_vectors(-1)
      { }

  void Register(OptionsItf *opts) {
    opts->Register("sample-frequency", &samp_freq,
                   "Waveform data sample frequency (must match the waveform file, "
                   "if specified there)");
    opts->Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
    opts->Register("frame-shift", &frame_shift_ms, "Frame shift in milliseconds");
    opts->Register("preemphasis-coefficient", &preemph_coeff,
                   "Coefficient for use in signal preemphasis");
    opts->Register("remove-dc-offset", &remove_dc_offset,
                   "Subtract mean from waveform on each frame");
    opts->Register("dither", &dither, "Dithering constant (0.0 means no dither). "
                   "If you turn this off, you should set the --energy-floor "
                   "option, e.g. to 1.0 or 0.1");
    opts->Register("window-type", &window_type, "Type of window "
                   "(\"hamming\"|\"hanning\"|\"povey\"|\"rectangular\""
                   "|\"sine\"|\"blackmann\")");
    opts->Register("blackman-coeff", &blackman_coeff,
                   "Constant coefficient for generalized Blackman window.");
    opts->Register("round-to-power-of-two", &round_to_power_of_two,
                   "If true, round window size to power of two by zero-padding "
                   "input to FFT.");
    opts->Register("snip-edges", &snip_edges,
                   "If true, end effects will be handled by outputting only frames that "
                   "completely fit in the file, and the number of frames depends on the "
                   "frame-length.  If false, the number of frames depends only on the "
                   "frame-shift, and we reflect the data at the ends.");
    opts->Register("allow-downsample", &allow_downsample,
                   "If true, allow the input waveform to have a higher frequency than "
                   "the specified --sample-frequency (and we'll downsample).");
    opts->Register("max-feature-vectors", &max_feature_vectors,
                   "Memory optimization. If larger than 0, periodically remove feature "
                   "vectors so that only this number of the latest feature vectors is "
                   "retained.");
    opts->Register("allow-upsample", &allow_upsample,
                   "If true, allow the input waveform to have a lower frequency than "
                   "the specified --sample-frequency (and we'll upsample).");
  }
  int32 WindowShift() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_shift_ms);
  }
  int32 WindowSize() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_length_ms);
  }
  int32 PaddedWindowSize() const {
    return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize()) :
                                    WindowSize());
  }
};

```

```cpp
struct FeatureWindowFunction {
  FeatureWindowFunction() {}
  explicit FeatureWindowFunction(const FrameExtractionOptions &opts);
  FeatureWindowFunction(const FeatureWindowFunction &other):
      window(other.window) { }
  Vector<BaseFloat> window;
};
//此函数返回可以从波形文件中提取给定样本数的帧数
// num_samples:波形文件中的样本数
// opts:帧提取配置
// flush:为true则使用全部语音；为false是online使用，或许还有更多语音，仅当opts.snips_edges==false时，才对答案有所不同。
//在在线解码的上下文中，一旦您知道（或决定）不再有数据输入，您将在结尾处使用flush == true调用它以清除所有剩余数据。
int32 NumFrames(int64 num_samples,const FrameExtractionOptions &opts,bool flush = true);
//此函数返回索引为“ frame”的帧的第一个样本的索引。 如果snip-edges = true，则只返回frame * opts.WindowShift（）;。
//如果snip-edges = false，则公式稍微复杂一点，结果可能为负。
int64 FirstSampleOfFrame(int32 frame,const FrameExtractionOptions &opts);
//给语音数据添加扰动
void Dither(VectorBase<BaseFloat> *waveform, BaseFloat dither_value);
//预加重
void Preemphasize(VectorBase<BaseFloat> *waveform, BaseFloat preemph_coeff);
//此功能在实际提取窗信号之后执行所有窗操作：根据配置，它会通过开窗功能进行抖动，直流偏移消除，预加重和乘法。
void ProcessWindow(const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   VectorBase<BaseFloat> *window,BaseFloat *log_energy_pre_window = NULL);
//提取波形的加窗后帧信息，包括由Process Windows（）完成的所有处理。
void ExtractWindow(int64 sample_offset,
                   const VectorBase<BaseFloat> &wave,
                   int32 f,
                   const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   Vector<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window = NULL);
```



## feature-fbank

fbank特征计算

```cpp
struct FbankOptions {
  FrameExtractionOptions frame_opts; //分帧选项，如帧长和帧移
  MelBanksOptions mel_opts;//滤波器个数
  bool use_energy;  // 特征的第一元素表示能量值
  BaseFloat energy_floor;//FBANK计算中的能量下限;仅在--use-energy = true时有所不同;
                         //仅在--dither = 0.0时才需要。 建议值：0.1或1.0“
  bool raw_energy;  // If true, 在预加重和加窗之前计算能量
  bool htk_compat;  // If true, put energy last (if using energy)
  bool use_log_fbank;  // if true (default), produce log-filterbank, else linear
  bool use_power;  // if true (default), 在滤波器组分析中使用power，否则使用幅度。

  FbankOptions(): mel_opts(23),
                 // defaults the #mel-banks to 23 for the FBANK computations.
                 // this seems to be common for 16khz-sampled data,
                 // but for 8khz-sampled data, 15 may be better.
                 use_energy(false),
                 energy_floor(0.0),
                 raw_energy(true),
                 htk_compat(false),
                 use_log_fbank(true),
                 use_power(true) {}

  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    mel_opts.Register(opts);
    opts->Register("use-energy", &use_energy,
                   "Add an extra dimension with energy to the FBANK output.");
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in FBANK computation. "
                   "Only makes a difference if --use-energy=true; only necessary if "
                   "--dither=0.0.  Suggested values: 0.1 or 1.0");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
    opts->Register("htk-compat", &htk_compat, "If true, put energy last.  "
                   "Warning: not sufficient to get HTK compatible features (need "
                   "to change other parameters).");
    opts->Register("use-log-fbank", &use_log_fbank,
                   "If true, produce log-filterbank, else produce linear.");
    opts->Register("use-power", &use_power,
                   "If true, use power, else use magnitude.");
  }
};
```

## feature-functions

feature-functions中包含的功能比较杂,主要功能如下

```cpp
//计算复数fft
void ComputePowerSpectrum(VectorBase<BaseFloat> *complex_fft);
//计算特征的导数
void ComputeDeltas(const DeltaFeaturesOptions &delta_opts,const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features);
//计算偏导数
void ComputeShiftedDeltas(const ShiftedDeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,Matrix<BaseFloat> *output_features);
//对feature进行扩展，类似于padding
void SpliceFrames(const MatrixBase<BaseFloat> &input_features,int32 left_context,
                  int32 right_context,Matrix<BaseFloat> *output_features);
//在时间上反转帧（用于向后解码）
void ReverseFrames(const MatrixBase<BaseFloat> &input_features,Matrix<BaseFloat> *output_features);
void InitIdftBases(int32 n_bases, int32 dimension, Matrix<BaseFloat> *mat_out);
//滑窗方式对特征进行归一化
void SlidingWindowCmn(const SlidingWindowCmnOptions &opts,
                      const MatrixBase<BaseFloat> &input,MatrixBase<BaseFloat> *output);
```

## feature-mfcc

计算mfcc特征

```cpp
struct MfccOptions {
  FrameExtractionOptions frame_opts;  //分帧选项，如帧长和帧移
  MelBanksOptions mel_opts;//滤波器个数
  int32 num_ceps;  // 倒谱个数 e.g. 13: num cepstral coeffs, counting zero.
  bool use_energy;  // 特征的第一元素表示能量值
  BaseFloat energy_floor;  // 默认为0;如果禁用dither，则设置为1.0或0.1
  bool raw_energy;  // If true, 在预加重和加窗之前计算能量
  BaseFloat cepstral_lifter;  // HTK兼容性的倒谱比例因子。 如果为0.0，则不进行提升.
  bool htk_compat;  // if true, put energy/C0 last and introduce a factor of
                    // sqrt(2) on C0 to be the same as HTK.
  MfccOptions() : mel_opts(23),
                  // defaults the #mel-banks to 23 for the MFCC computations.
                  // this seems to be common for 16khz-sampled data,
                  // but for 8khz-sampled data, 15 may be better.
                  num_ceps(13),
                  use_energy(true),
                  energy_floor(0.0),
                  raw_energy(true),
                  cepstral_lifter(22.0),
                  htk_compat(false) {}

  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    mel_opts.Register(opts);
    opts->Register("num-ceps", &num_ceps,
                   "Number of cepstra in MFCC computation (including C0)");
    opts->Register("use-energy", &use_energy,
                   "Use energy (not C0) in MFCC computation");
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in MFCC computation. "
                   "Only makes a difference if --use-energy=true; only necessary if "
                   "--dither=0.0.  Suggested values: 0.1 or 1.0");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
    opts->Register("cepstral-lifter", &cepstral_lifter,
                   "Constant that controls scaling of MFCCs");
    opts->Register("htk-compat", &htk_compat,
                   "If true, put energy or C0 last and use a factor of sqrt(2) on "
                   "C0.  Warning: not sufficient to get HTK compatible features "
                   "(need to change other parameters).");
  }
};
```

## feature-plp

计算plp特征

```cpp
struct PlpOptions {
  FrameExtractionOptions frame_opts;  //分帧选项，如帧长和帧移
  MelBanksOptions mel_opts; //滤波器个数
  int32 lpc_order;  
  int32 num_ceps;  // 倒谱个数
  bool use_energy;  // 特征的第一元素表示能量值
  BaseFloat energy_floor; // 默认为0;如果禁用dither，则设置为1.0或0.1
  bool raw_energy;  //If true, 在预加重和加窗之前计算能量
  BaseFloat compress_factor;
  int32 cepstral_lifter;
  BaseFloat cepstral_scale;
  bool htk_compat;  // if true, put energy/C0 last and introduce a factor of
                    // sqrt(2) on C0 to be the same as HTK.

  PlpOptions() : mel_opts(23),
                 // default number of mel-banks for the PLP computation; this
                 // seems to be common for 16kHz-sampled data. For 8kHz-sampled
                 // data, 15 may be better.
                 lpc_order(12),
                 num_ceps(13),
                 use_energy(true),
                 energy_floor(0.0),
                 raw_energy(true),
                 compress_factor(0.33333),
                 cepstral_lifter(22),
                 cepstral_scale(1.0),
                 htk_compat(false) {}

  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    mel_opts.Register(opts);
    opts->Register("lpc-order", &lpc_order,
                   "Order of LPC analysis in PLP computation");
    opts->Register("num-ceps", &num_ceps,
                   "Number of cepstra in PLP computation (including C0)");
    opts->Register("use-energy", &use_energy,
                   "Use energy (not C0) for zeroth PLP feature");
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in PLP computation. "
                   "Only makes a difference if --use-energy=true; only necessary if "
                   "--dither=0.0.  Suggested values: 0.1 or 1.0");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
    opts->Register("compress-factor", &compress_factor,
                   "Compression factor in PLP computation");
    opts->Register("cepstral-lifter", &cepstral_lifter,
                   "Constant that controls scaling of PLPs");
    opts->Register("cepstral-scale", &cepstral_scale,
                   "Scaling constant in PLP computation");
    opts->Register("htk-compat", &htk_compat,
                   "If true, put energy or C0 last.  Warning: not sufficient "
                   "to get HTK compatible features (need to change other "
                   "parameters).");
  }
};
```

## feature-spectrogram

计算频谱图特征

```cpp
struct SpectrogramOptions {
  FrameExtractionOptions frame_opts; //分帧选项，如帧长和帧移
  BaseFloat energy_floor;
  bool raw_energy;  //If true, 在预加重和加窗之前计算能量
  bool return_raw_fft; // 如果为true，则返回原始FFT频谱。在这种情况下，Dim()将返回预期尺寸的两倍（因为它的复数域）

  SpectrogramOptions() :
    energy_floor(0.0),
    raw_energy(true),
    return_raw_fft(false) {}

  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in Spectrogram "
                   "computation.  Caution: this floor is applied to the zeroth "
                   "component, representing the total signal energy.  The "
                   "floor on the individual spectrogram elements is fixed at "
                   "std::numeric_limits<float>::epsilon().");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
    opts->Register("return-raw-fft", &return_raw_fft,
                   "If true, return raw FFT complex numbers instead of log magnitudes");
  }
};
```

## online-feature

在线特征提取，解决语音过长的情况下离线特征提取占用内存过大的问题。

### RecyclingVector

此类用作特征向量的存储，并带有通过删除旧元素来限制内存使用的选项。 删除的帧索引被“记住”，因此，无论MAX_ITEMS设置如何，用户始终提供索引，就好像没有执行删除操作一样。 这在处理很长的记录时很有用，否则在不删除功能时会导致内存最终崩溃。

```cpp
class RecyclingVector {
public:
  /// By default it does not remove any elements.
  RecyclingVector(int items_to_hold = -1);
  /// The ownership is being retained by this collection - do not delete the item.
  Vector<BaseFloat> *At(int index) const;
  /// The ownership of the item is passed to this collection - do not delete the item.
  void PushBack(Vector<BaseFloat> *item);
  /// This method returns the size as if no "recycling" had happened,
  /// i.e. equivalent to the number of times the PushBack method has been called.
  int Size() const;
  ~RecyclingVector();
private:
  std::deque<Vector<BaseFloat>*> items_;   //双端队列，用于存放特征
  int items_to_hold_;
  int first_available_index_;
};
```

### OnlineGenericBaseFeature

这是用于在线特征提取的模板类。以MfccComputer、FbankComputer、PlpComputer进行基本特征提取

```cpp
template<class C>
class OnlineGenericBaseFeature: public OnlineBaseFeature {
 public:
  virtual int32 Dim() const { return computer_.Dim(); }
  // Note: IsLastFrame() will only ever return true if you have called
  // InputFinished() (and this frame is the last frame).
  virtual bool IsLastFrame(int32 frame) const {
    return input_finished_ && frame == NumFramesReady() - 1;
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return computer_.GetFrameOptions().frame_shift_ms / 1000.0f;
  }
  virtual int32 NumFramesReady() const { return features_.Size(); }
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  // Constructor from options class
  explicit OnlineGenericBaseFeature(const typename C::Options &opts);
  // 有新的语音数据时会从应用程序调用
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);


  // 不再提供任何波形
  virtual void InputFinished();

 private:
  // 从waveform_remainder_取语音，计算特征存放到features_，并从waveform_remainder_移除语音
  // 同时将waveform_offset_递增相同的量
  void ComputeFeatures();

  void MaybeCreateResampler(BaseFloat sampling_rate);

  C computer_;  // MFCC or PLP or filterbank computation

  // 如果输入采样频率不等于预期采样率，则重新采样
  std::unique_ptr<LinearResample> resampler_;
  
  FeatureWindowFunction window_function_;

  // feature_已经计算出的Mfcc/Plp/Fbank.
  RecyclingVector features_;

  // True if the user has called "InputFinished()"
  bool input_finished_;

  // 配置文件中的采样率
  BaseFloat sampling_frequency_;

  // waveform_offset_是我们已经舍弃的波形样本数，即在'waveform_remainder_'之前的样本数。
  int64 waveform_offset_;
  // waveform_remainder_是一小段波形，提取完所有可能的帧后可能需要保留
  Vector<BaseFloat> waveform_remainder_;
};
```

```cpp
typedef OnlineGenericBaseFeature<MfccComputer> OnlineMfcc;
typedef OnlineGenericBaseFeature<PlpComputer> OnlinePlp;
typedef OnlineGenericBaseFeature<FbankComputer> OnlineFbank;
```

### OnlineMatrixFeature

对已经离线提取的音频特征转为在线提取的音频特征

```cpp
class OnlineMatrixFeature: public OnlineFeatureInterface {
 public:
  /// Caution: this class maintains the const reference from the constructor, so
  /// don't let it go out of scope while this object exists.
  explicit OnlineMatrixFeature(const MatrixBase<BaseFloat> &mat): mat_(mat) { }
  virtual int32 Dim() const { return mat_.NumCols(); }
  virtual BaseFloat FrameShiftInSeconds() const {
    return 0.01f;
  }
  virtual int32 NumFramesReady() const { return mat_.NumRows(); }
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
    feat->CopyFromVec(mat_.Row(frame));
  }
  virtual bool IsLastFrame(int32 frame) const {
    return (frame + 1 == mat_.NumRows());
  }
 private:
  const MatrixBase<BaseFloat> &mat_;
};
```

### OnlineCmvnOptions

与SlidingWindowCmnOptions相似，但也有区别。 它是在OnlineCmn中，如果没有前一次语音可用，则从全局统计数据中继承数据，或者，如果以前的语音可用，但数据总量少于prev_frames，我们从全局统计信息中填充最多“ global_frames”帧。

```cpp
struct OnlineCmvnOptions {
  int32 cmn_window;
  int32 speaker_frames;  // must be <= cmn_window
  int32 global_frames;  // must be <= speaker_frames.
  bool normalize_mean;  // Must be true if normalize_variance==true.
  bool normalize_variance;
  int32 modulus;  // not configurable from command line, relates to how the
                  // class computes the cmvn internally.  smaller->more
                  // time-efficient but less memory-efficient.  Must be >= 1.
  int32 ring_buffer_size;  // not configurable from command line; size of ring
                           // buffer used for caching CMVN stats.  Must be >=
                           // modulus.
  std::string skip_dims; // Colon-separated list of dimensions to skip normalization
                         // of, e.g. 13:14:15.
  OnlineCmvnOptions():
      cmn_window(600),
      speaker_frames(600),
      global_frames(200),
      normalize_mean(true),
      normalize_variance(false),
      modulus(20),
      ring_buffer_size(20),
      skip_dims("") { }

  void Check() const {
    KALDI_ASSERT(speaker_frames <= cmn_window && global_frames <= speaker_frames
                 && modulus > 0);
  }

  void Register(ParseOptions *po) {
    po->Register("cmn-window", &cmn_window, "Number of frames of sliding "
                 "context for cepstral mean normalization.");
    po->Register("global-frames", &global_frames, "Number of frames of "
                 "global-average cepstral mean normalization stats to use for "
                 "first utterance of a speaker");
    po->Register("speaker-frames", &speaker_frames, "Number of frames of "
                 "previous utterance(s) from this speaker to use in cepstral "
                 "mean normalization");
    // we name the config string "norm-vars" for compatibility with
    // ../featbin/apply-cmvn.cc
    po->Register("norm-vars", &normalize_variance, "If true, do "
                 "cepstral variance normalization in addition to cepstral mean "
                 "normalization ");
    po->Register("norm-means", &normalize_mean, "If true, do mean normalization "
                 "(note: you cannot normalize the variance but not the mean)");
    po->Register("skip-dims", &skip_dims, "Dimensions to skip normalization of "
                 "(colon-separated list of integers)");}
};
```

### OnlineCmvnState

OnlineCmvnState存储发声之间的CMVN适应状态

### OnlineCmvn

此类进行倒谱均值和方差的在线版本，通常仅在基于“在线” GMM的解码中执行此操作

```cpp
class OnlineCmvn: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return src_->Dim(); }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }
  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  /// 设置cmvn状态的初始化程序。 
  //如果您以前没有同一说话者的讲话，则应该从某些全局CMVN统计信息中初始化CMVN状态，
  //这可以通过使用“和矩阵”将训练数据中的所有cmvn统计信息相加而得出。 
  //这只是在文件开头提供了一个合理的起点。 
  //如果确实有来自同一说话者或至少相似环境的先前讲话，则应通过从先前讲话中调用GetState对其进行初始化。
  OnlineCmvn(const OnlineCmvnOptions &opts,
             const OnlineCmvnState &cmvn_state,
             OnlineFeatureInterface *src);

  /// 不设置cmvn状态的初始化程序：调用此方法后，应调用SetState（）。
  OnlineCmvn(const OnlineCmvnOptions &opts,
             OnlineFeatureInterface *src);

  // 从该语音输出任何状态信息到“ cmvn_state”。 调用之前的“ cmvn_state”值无关紧要：
  // 输出取决于类初始化所使用的OnlineCmvnState的值，输入特征值直至cur_frame以及用户调用Freeze（）的效果。 
  // 如果cur_frame为-1，它将仅输出提供给该对象的未修改的原始状态。
  void GetState(int32 cur_frame,
                OnlineCmvnState *cmvn_state);

  // 该函数可用于从外部修改CMVN计算的状态，但只能在处理任何数据之前调用它（否则它将崩溃）。 
  //这种“状态”实际上只是在发声之间传播的信息，而不是发声内部的计算状态。
  void SetState(const OnlineCmvnState &cmvn_state);

  //冻结CMN到在帧“ cur_frame”处测量时的状态，并且将阻止其进一步更改。如果在先前的帧上调用GetFrame（），
  //它将使用cur_frame中的CMVN统计信息。如果随后调用OutputState并使用此状态初始化下一个话语的CMVN对象，则它也适用于将来。
  void Freeze(int32 cur_frame);

  virtual ~OnlineCmvn();
 private:

  /// 使CMVN统计信息（以2 x（dim + 1）矩阵的常规格式存储）变得平滑
  static void SmoothOnlineCmvnStats(const MatrixBase<double> &speaker_stats,
                                    const MatrixBase<double> &global_stats,
                                    const OnlineCmvnOptions &opts,
                                    MatrixBase<double> *stats);

  /// 获取CMVN统计信息的最新缓存帧。 [如果没有缓存任何帧，为第零帧设置空统计信息并返回]。
  void GetMostRecentCachedFrame(int32 frame,
                                int32 *cached_frame,
                                MatrixBase<double> *stats);

  /// 缓存帧信息
  void CacheFrame(int32 frame, const MatrixBase<double> &stats);

  /// 初始化用于缓存统计信息的环形缓冲区。
  inline void InitRingBufferIfNeeded();

  /// 计算该帧的原始CMVN统计信息，并利用raw_stats_中的缓存统计信息（并在必要时进行更新）。 
  // 这表示最后opts_.cmn_window帧的（x，x ^ 2，count）个统计信息。
  void ComputeStatsForFrame(int32 frame,
                            MatrixBase<double> *stats);


  OnlineCmvnOptions opts_;
  std::vector<int32> skip_dims_; // Skip CMVN for these dimensions.  Derived from opts_.
  OnlineCmvnState orig_state_;   // 保存要计算当前语音之前的状态
  Matrix<double> frozen_state_;  // 如果用户调用Freeze（），则此变量将反映我们冻结的CMVN状态。

  // 输入的原始统计信息
  std::vector<Matrix<double>*> cached_stats_modulo_;
  // 缓存统计信息的环形缓冲区。
  std::vector<std::pair<int32, Matrix<double> > > cached_stats_ring_;
  // 临时变量.
  Matrix<double> temp_stats_;
  Vector<BaseFloat> temp_feats_;
  Vector<double> temp_feats_dbl_;

  OnlineFeatureInterface *src_;  // Not owned here
};
```

### OnlineSpliceFrames

在线解码时用到的拼帧类

```cpp
class OnlineSpliceFrames: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const {
    return src_->Dim() * (1 + left_context_ + right_context_);
  }
  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  // functions that are not in the interface.
  OnlineSpliceFrames(const OnlineSpliceOptions &opts,
                     OnlineFeatureInterface *src):
      left_context_(opts.left_context), right_context_(opts.right_context),
      src_(src) { }
 private:
  int32 left_context_;
  int32 right_context_;
  OnlineFeatureInterface *src_;  // Not owned here
};
```

### OnlineTransform

在线仿射或线性变换

```cpp
class OnlineTransform: public OnlineFeatureInterface {
 public:
  // First, functions that are present in the interface:
  virtual int32 Dim() const { return offset_.Dim(); }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void GetFrames(const std::vector<int32> &frames,
                         MatrixBase<BaseFloat> *feats);

  // Next, functions that are not in the interface.
  OnlineTransform(const MatrixBase<BaseFloat> &transform,
                  OnlineFeatureInterface *src);
 private:
  OnlineFeatureInterface *src_;  // Not owned here
  Matrix<BaseFloat> linear_term_;
  Vector<BaseFloat> offset_;
};
```

### OnlineDeltaFeature

在线特征差分

```cpp
class OnlineDeltaFeature: public OnlineFeatureInterface {
 public:
  // First, functions that are present in the interface:
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  // Next, functions that are not in the interface.
  OnlineDeltaFeature(const DeltaFeaturesOptions &opts,
                     OnlineFeatureInterface *src);
 private:
  OnlineFeatureInterface *src_;  // Not owned here
  DeltaFeaturesOptions opts_;
  DeltaFeatures delta_features_;  // This class contains just a few
                                  // coefficients.
};
```

### OnlineCacheFeature

在线特征缓存

```cpp
class OnlineCacheFeature: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return src_->Dim(); }
  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }
  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  virtual void GetFrames(const std::vector<int32> &frames,
                         MatrixBase<BaseFloat> *feats);
  virtual ~OnlineCacheFeature() { ClearCache(); }
  // Things that are not in the shared interface:
  void ClearCache();  // this should be called if you change the underlying
                      // features in some way.
  explicit OnlineCacheFeature(OnlineFeatureInterface *src): src_(src) { }
 private:
  OnlineFeatureInterface *src_;  // Not owned here
  std::vector<Vector<BaseFloat>* > cache_;
};
```

### OnlineAppendFeature

在线将两个流拼接为一个流

```cpp
class OnlineAppendFeature: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return src1_->Dim() + src2_->Dim(); }
  virtual bool IsLastFrame(int32 frame) const {
    return (src1_->IsLastFrame(frame) || src2_->IsLastFrame(frame));
  }
  // Hopefully sources have the same rate
  virtual BaseFloat FrameShiftInSeconds() const {
    return src1_->FrameShiftInSeconds();
  }
  virtual int32 NumFramesReady() const {
    return std::min(src1_->NumFramesReady(), src2_->NumFramesReady());
  }
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  virtual ~OnlineAppendFeature() {  }
  OnlineAppendFeature(OnlineFeatureInterface *src1,
      OnlineFeatureInterface *src2): src1_(src1), src2_(src2) { }
 private:
  OnlineFeatureInterface *src1_;
  OnlineFeatureInterface *src2_;
};
```

### online特征提取示例说明

代码路径：src/feat/online-feature-test.cc

以mfcc的online特征提取为例

```cpp
//创建online_mfcc,其中包括
//Vector<BaseFloat> waveform_remainder_;  保存未进行特征提取的原始语音
//int64 waveform_offset_;   已经进行特征提取的原始语音位置
//bool input_finished_;     特征提取结束标志位
// MfccComputer computer_   mfcc特征提取
//RecyclingVector features_;  存放提取后的mfcc特征，可以自定义循环队列的大小
OnlineMfcc online_mfcc(op); 
//接收音频，并将音频追加到waveform_remainder_中
//然后调用ComputeFeatures()计算当前waveform_remainder_音频特征
online_mfcc.AcceptWaveform(wave.SampFreq(), wave_piece);
//计算当前waveform_remainder_音频特征
template <class C>
void OnlineGenericBaseFeature<C>::ComputeFeatures() {
  const FrameExtractionOptions &frame_opts = computer_.GetFrameOptions();
  int64 num_samples_total = waveform_offset_ + waveform_remainder_.Dim();  //语音总样本数
  int32 num_frames_old = features_.Size(),  
    //计算waveform_remainder_中语音有多少帧
      num_frames_new = NumFrames(num_samples_total, frame_opts,  
                                 input_finished_);
  KALDI_ASSERT(num_frames_new >= num_frames_old);

  Vector<BaseFloat> window;
  bool need_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 frame = num_frames_old; frame < num_frames_new; frame++) {
    BaseFloat raw_log_energy = 0.0;
    //对当前帧执行执行加窗和预加重等操作，
    //waveform_offset_表示在整个语音中当前帧的偏移量，而不是在waveform_remainder_中的偏移量
    //特征提取完成后，语音会从waveform_remainder_中删除，以节省内存
    ExtractWindow(waveform_offset_, waveform_remainder_, frame,
                  frame_opts, window_function_, &window,
                  need_raw_log_energy ? &raw_log_energy : NULL);
    Vector<BaseFloat> *this_feature = new Vector<BaseFloat>(computer_.Dim(),
                                                            kUndefined);
    // note: this online feature-extraction code does not support VTLN.
    BaseFloat vtln_warp = 1.0;
    //计算mfcc特征
    computer_.Compute(raw_log_energy, vtln_warp, &window, this_feature);
    features_.PushBack(this_feature);
  }
  // 返回帧num_frames_new的起始语音偏移
  int64 first_sample_of_next_frame = FirstSampleOfFrame(num_frames_new,
                                                        frame_opts);
  //计算提取完特征的语音采样数
  int32 samples_to_discard = first_sample_of_next_frame - waveform_offset_;
  //从waveform_remainder_中删除提取完特征的语音
  if (samples_to_discard > 0) {
    // discard the leftmost part of the waveform that we no longer need.
    int32 new_num_samples = waveform_remainder_.Dim() - samples_to_discard;
    if (new_num_samples <= 0) {
      // odd, but we'll try to handle it.
      waveform_offset_ += waveform_remainder_.Dim();
      waveform_remainder_.Resize(0);
    } else {
      Vector<BaseFloat> new_remainder(new_num_samples);
      new_remainder.CopyFromVec(waveform_remainder_.Range(samples_to_discard,
                                                          new_num_samples));
      waveform_offset_ += samples_to_discard;
      waveform_remainder_.Swap(&new_remainder);
    }
  }
}
//停止接收新语音，计算waveform_remainder_中所有语音的特征
online_mfcc.InputFinished();
```



## pitch-functions

计算基频特征

### PitchExtractionOptions

提取pitch选项

```cpp
struct PitchExtractionOptions {
  // FrameExtractionOptions frame_opts;
  BaseFloat samp_freq;          // 采样率
  BaseFloat frame_shift_ms;     // 帧移，in milliseconds.
  BaseFloat frame_length_ms;    // 帧长，in milliseconds.
  BaseFloat preemph_coeff;      // 预加重系数. [use is deprecated.]
  BaseFloat min_f0;             // 最小f0搜索 (Hz)
  BaseFloat max_f0;             // 最大f0搜索 (Hz)
  BaseFloat soft_min_f0;        // 以软方式应用的最小值f0，不得超过min-f0
  BaseFloat penalty_factor;     // FO变化的成本因子
  BaseFloat lowpass_cutoff;     // 低通滤波器的截止频率
  BaseFloat resample_freq;      // 在对NCCF进行上采样时确定滤波器宽度的整数
  BaseFloat delta_pitch;        // pitch剪枝的容忍度
  BaseFloat nccf_ballast;       // 增加此系数会降低静音帧的NCCF，有助于确保清音区域的音高连续性
  int32 lowpass_filter_width;   // 确定低通滤波器的滤波器宽度的整数
  int32 upsample_filter_width;  // 在对NCCF进行上采样时确定滤波器宽度的整数

  // 以下是与在线音高提取算法有关的较新的配置变量，这些变量在原始论文中没有提供。

  // 在线操作中，我们允许音高处理引入的最大延迟帧数。 
  //如果将此值设置为较大的值，则维特比回溯不会有任何不准确之处（但它可能会让您等待查看音高）。 
  //与在线操作不太相关：normalization-right-context更为相关，您可以将该值保留为零。
  int32 max_frames_latency;

  // 仅与compute-kaldi-pitch-feats调用的功能ComputeKaldiPitch有关。 
  //如果非零，则将输入作为此大小的块提供。 这会影响能量归一化，这对所得功能（尤其是在文件开头）的影响很小。 
  //为了获得与在线操作的最佳兼容性（例如，如果您打算为在线解码设置训练模型），可能需要将此值设置为一个较小的值，例如一帧。
  int32 frames_per_chunk;

  // 仅与compute-kaldi-pitch-feats调用的ComputeKaldiPitch函数相关，并且仅当frames_per_chunk为非零时才相关。 
  //如果为true，它将在features可用时立即对其进行查询，这将模拟您在在线解码中获得的first-pass features。 
  //如果为false，则在调用InputFinished（）之后，您将获得的features与话语末尾可用的features相同： 在晶格记录中。
  bool simulate_first_pass_online;

  // 仅与在线操作或模拟在线操作有关（例如，在设置frames_per_chunk时）。 
  //重新计算NCCF的帧索引（例如，帧索引500 = 5秒后）；
  //会在长话语结束时引入不必要的延迟，而没有什么好处。
  int32 recompute_frame;

  // 这是仅用于测试在线音高提取的“隐藏配置”。 
  //如果为true，我们将计算镇流项的信号均方根值，直到当前帧为止，而不是当前信号块的末尾。 
  //这使得输出对分块不敏感，这对于测试目的很有用。
  bool nccf_ballast_online;
  bool snip_edges;
  PitchExtractionOptions():
      samp_freq(16000),
      frame_shift_ms(10.0),
      frame_length_ms(25.0),
      preemph_coeff(0.0),
      min_f0(50),
      max_f0(400),
      soft_min_f0(10.0),
      penalty_factor(0.1),
      lowpass_cutoff(1000),
      resample_freq(4000),
      delta_pitch(0.005),
      nccf_ballast(7000),
      lowpass_filter_width(1),
      upsample_filter_width(5),
      max_frames_latency(0),
      frames_per_chunk(0),
      simulate_first_pass_online(false),
      recompute_frame(500),
      nccf_ballast_online(false),
      snip_edges(true) { }
};
```

### ProcessPitchOptions

pitch特征后处理选项

```cpp
struct ProcessPitchOptions {
  BaseFloat pitch_scale;  // 最终的normalized-log-pitch特征将以此值缩放
  BaseFloat pov_scale;    // the final POV feature is scaled with this value
  BaseFloat pov_offset;   // An offset that can be added to the final POV
                          // feature (useful for online-decoding, where we don't
                          // do CMN to the pitch-derived features.

  BaseFloat delta_pitch_scale;
  BaseFloat delta_pitch_noise_stddev;  //添加到delta-pitch的噪声的标准差
  int32 normalization_left_context;    // 左上下文用于滑动窗口归一化
  int32 normalization_right_context;   // 在线解码中应减少此设置以减少延迟
  int32 delta_window;
  int32 delay;
  bool add_pov_feature;
  bool add_normalized_log_pitch;
  bool add_delta_pitch;
  bool add_raw_log_pitch;

  ProcessPitchOptions() :
      pitch_scale(2.0),
      pov_scale(2.0),
      pov_offset(0.0),
      delta_pitch_scale(10.0),
      delta_pitch_noise_stddev(0.005),
      normalization_left_context(75),
      normalization_right_context(75),
      delta_window(2),
      delay(0),
      add_pov_feature(true),
      add_normalized_log_pitch(true),
      add_delta_pitch(true),
      add_raw_log_pitch(false) { }
};
```

### OnlinePitchFeature

在线计算pitch特征

```cpp
class OnlinePitchFeature: public OnlineBaseFeature {
 public:
  explicit OnlinePitchFeature(const PitchExtractionOptions &opts);
  virtual int32 Dim() const { return 2; /* (NCCF, pitch) */ }
  virtual int32 NumFramesReady() const;
  virtual BaseFloat FrameShiftInSeconds() const;
  virtual bool IsLastFrame(int32 frame) const;
  /// Outputs the two-dimensional feature consisting of (pitch, NCCF).  You
  /// should probably post-process this using class OnlineProcessPitch.
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);
  virtual void InputFinished();
  virtual ~OnlinePitchFeature();
 private:
  OnlinePitchFeatureImpl *impl_;
};
```

### OnlineProcessPitch

对pitch特征进一步处理

```cpp
class OnlineProcessPitch: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return dim_; }
  virtual bool IsLastFrame(int32 frame) const {
    if (frame <= -1)
      return src_->IsLastFrame(-1);
    else if (frame < opts_.delay)
      return src_->IsLastFrame(-1) == true ? false : src_->IsLastFrame(0);
    else
      return src_->IsLastFrame(frame - opts_.delay);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  virtual ~OnlineProcessPitch() {  }
  // Does not take ownership of "src".
  OnlineProcessPitch(const ProcessPitchOptions &opts,
                     OnlineFeatureInterface *src);
 private:
  enum { kRawFeatureDim = 2};  // anonymous enum to define a constant.
                               // kRawFeatureDim defines the dimension
                               // of the input: (nccf, pitch)
  ProcessPitchOptions opts_;
  OnlineFeatureInterface *src_;
  int32 dim_;  // Output feature dimension, set in initializer.
  struct NormalizationStats {
    int32 cur_num_frames;      // value of src_->NumFramesReady() when
                               // "mean_pitch" was set.
    bool input_finished;       // true if input data was finished when
                               // "mean_pitch" was computed.
    double sum_pov;            // sum of pov over relevant range
    double sum_log_pitch_pov;  // sum of log(pitch) * pov over relevant range

    NormalizationStats(): cur_num_frames(-1), input_finished(false),
                          sum_pov(0.0), sum_log_pitch_pov(0.0) { }
  };

  std::vector<BaseFloat> delta_feature_noise_;
  std::vector<NormalizationStats> normalization_stats_;
  /// Computes and returns the POV feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetPovFeature(int32 frame) const;
  /// Computes and returns the delta-log-pitch feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetDeltaPitchFeature(int32 frame);
  /// Computes and returns the raw log-pitch feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetRawLogPitchFeature(int32 frame) const;
  /// Computes and returns the mean-subtracted log-pitch feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetNormalizedLogPitchFeature(int32 frame);
  /// Computes the normalization window sizes.
  inline void GetNormalizationWindow(int32 frame,
                                     int32 src_frames_ready,
                                     int32 *window_begin,
                                     int32 *window_end) const;
  /// Makes sure the entry in normalization_stats_ for this frame is up to date;
  /// called from GetNormalizedLogPitchFeature.
  inline void UpdateNormalizationStats(int32 frame);
};
```

```cpp
void ComputeKaldiPitch(const PitchExtractionOptions &opts,
                       const VectorBase<BaseFloat> &wave,
                       Matrix<BaseFloat> *output);
void ProcessPitch(const ProcessPitchOptions &opts,
                  const MatrixBase<BaseFloat> &input,
                  Matrix<BaseFloat> *output);
void ComputeAndProcessKaldiPitch(const PitchExtractionOptions &pitch_opts,
                                 const ProcessPitchOptions &process_opts,
                                 const VectorBase<BaseFloat> &wave,
                                 Matrix<BaseFloat> *output);
```

## resample

对音频进行重采样。包括ArbitraryResample和LinearResample

### ArbitraryResample

类ArbitraryResample允许您在任意指定的时间值（不必线性间隔）上对信号进行重新采样（假定在采样区域外为零，不是周期性的）。 低通滤波器的截止值“ filter_cutoff_hz”应小于采样率的一半。 “ num_zeros”应至少为两个，最好大于两个； 数字越大，过滤器越锐利，但效率较低。

```cpp
class ArbitraryResample {
 public:
  ArbitraryResample(int32 num_samples_in,
                    BaseFloat samp_rate_hz,
                    BaseFloat filter_cutoff_hz,
                    const Vector<BaseFloat> &sample_points_secs,
                    int32 num_zeros);
  int32 NumSamplesIn() const { return num_samples_in_; }
  int32 NumSamplesOut() const { return weights_.size(); }
  /// 批量 resampling，每行表示一个语音数据.
  /// input.NumRows() == output.NumRows() 并且不为0
  /// input.NumCols() == NumSamplesIn()
  /// and output.NumCols() == NumSamplesOut().
  void Resample(const MatrixBase<BaseFloat> &input,
                MatrixBase<BaseFloat> *output) const;

  /// 单条语音resample
  void Resample(const VectorBase<BaseFloat> &input,
                VectorBase<BaseFloat> *output) const;
 private:
  void SetIndexes(const Vector<BaseFloat> &sample_points);
  void SetWeights(const Vector<BaseFloat> &sample_points);
  BaseFloat FilterFunc(BaseFloat t) const;
  int32 num_samples_in_;
  BaseFloat samp_rate_in_;
  BaseFloat filter_cutoff_;
  int32 num_zeros_;

  std::vector<int32> first_index_;  // 对该输出样本索引求和的第一个输入样本索引。
  std::vector<Vector<BaseFloat> > weights_;
};
```

### LinearResample

LinearResample是ArbitraryResample的特例，我们要以线性间隔对信号进行重采样（这意味着我们要对信号进行上采样或下采样）。 它比ArbitraryResample更有效，因为我们只能构造一次。

我们要求将输入和输出采样率指定为整数，因为这是指定其比率合理的简单方法。

```cpp
class LinearResample {
 public:
  /// 输入和输出采样率为整数,filter_cutoff_hz必须小于samp_rate_in_hz / 2且小于samp_rate_out_hz / 2。
  //num_zeros控制滤镜的清晰度,num_zeros越大清晰，但效率更低。建议使用4到10
  LinearResample(int32 samp_rate_in_hz,
                 int32 samp_rate_out_hz,
                 BaseFloat filter_cutoff_hz,
                 int32 num_zeros);

  /// 此功能进行重采样。 
  //如果使用flush == true，而从未使用flush == false调用它，则它只是对输入信号进行重新采样（它将输出调整为适当数量的采样）。
  ///
  /// You can also use this function to process a signal a piece at a time.
  /// suppose you break it into piece1, piece2, ... pieceN.  You can call
  /// \code{.cc}
  /// Resample(piece1, &output1, false);
  /// Resample(piece2, &output2, false);
  /// Resample(piece3, &output3, true);
  /// \endcode
  /// 如果使用flush == false调用它，它将不会输出最后几个样本，但会记住它们，因此，如果以后再给它第二个输入信号，
  //它就可以正确处理。 如果您最近对该对象的调用是flush == false，则它将具有内部状态；否则，它将处于内部状态。 
  //您可以通过调用Reset（）删除它。 空输入是可以接受的。
  void Resample(const VectorBase<BaseFloat> &input,
                bool flush,
                Vector<BaseFloat> *output);

  /// 调用函数Reset（）会在处理新信号之前重置对象的状态。 
  //仅当您对某个信号调用了Resample（x，y，false）时，才有必要。在信号之间不必要地调用它不会造成任何伤害。
  void Reset();

  //// Return the input and output sampling rates (for checks, for example)
  inline int32 GetInputSamplingRate() { return samp_rate_in_; }
  inline int32 GetOutputSamplingRate() { return samp_rate_out_; }
 private:
  /// 返回输出信号的采样数，
  //如果flush == true，则返回最大的n使（n / samp_rate_out_）在[0，input_num_samp / samp_rate_in_）区间
  //如果flush == false,返回最大的n，使得（n/samp_rate_out_）在[0，input_num_samp/samp_rate_in_-window_width）区间
  //window_width定义为num_zeros /（2.0 * filter_cutoff_）
  int64 GetNumOutputSamples(int64 input_num_samp, bool flush) const;
  /// samp_out:输出索引 
  /// first_samp_in:第一个有权重的输入信号索引,
  /// samp_out_wrapped:输入上所有权重的打包
  inline void GetIndexes(int64 samp_out,
                         int64 *first_samp_in,
                         int32 *samp_out_wrapped) const;
  void SetRemainder(const VectorBase<BaseFloat> &input);
  void SetIndexesAndWeights();
  BaseFloat FilterFunc(BaseFloat) const;
  // The following variables are provided by the user.
  int32 samp_rate_in_;
  int32 samp_rate_out_;
  BaseFloat filter_cutoff_;
  int32 num_zeros_;
  int32 input_samples_in_unit_;   //最小重复单位中的输入样本数：
                               //num_samp_in_ = samp_rate_in_hz / Gcd（samp_rate_in_hz，samp_rate_out_hz）
  int32 output_samples_in_unit_;  //最小重复单位的输出采样数：
                               //num_samp_out_ = samp_rate_out_hz / Gcd（samp_rate_in_hz，samp_rate_out_hz）

  /// 我们对该输出样本索引求和的第一个输入样本索引。 可能是负面的； 
  //开头的任何截断都将单独处理。 这仅用于前几个输出样本，但是我们可以推断任意输出样本的正确输入样本索引.
  std::vector<int32> first_index_;

  /// Weights on the input samples, for this output-sample index.
  std::vector<Vector<BaseFloat> > weights_;

  // the following variables keep track of where we are in a particular signal,
  // if it is being provided over multiple calls to Resample().

  int64 input_sample_offset_;  ///< The number of input samples we have
                               ///< already received for this signal
                               ///< (including anything in remainder_)
  int64 output_sample_offset_;  ///< The number of samples we have already
                                ///< output for this signal.
  Vector<BaseFloat> input_remainder_;  ///< A small trailing part of the
                                       ///< previously seen input signal.
};
```

```cpp
//下采样或上采样波形。 这是“ LinearResample”类的包装。
void ResampleWaveform(BaseFloat orig_freq, const VectorBase<BaseFloat> &wave,
                      BaseFloat new_freq, Vector<BaseFloat> *new_wave);
```

## signal

使用三种方法实现了对两个信号的卷积

```cpp
//该函数实现了两个信号的简单的非基于FFT的卷积。 建议使用更高效的基于FFT的卷积函数。
void ConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal);
//该功能实现了两个信号的基于FFT的卷积。 但是，这应该是BlockConvolveSignals（）的低效率版本，因为它使用单个FFT处理整个信号。
void FFTbasedConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal);
//此功能使用重叠相加方法实现两个信号的基于FFT的块卷积。 这是一种使用有限脉冲响应滤波器评估长信号离散卷积的有效方法。
void FFTbasedBlockConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal);
```

## wave-reader

读写wav文件

### WaveInfo

wav文件头信息

```cpp
class WaveInfo {
 public:
  WaveInfo() : samp_freq_(0), samp_count_(0),
               num_channels_(0), reverse_bytes_(0) {}

  /// Is stream size unknown? Duration and SampleCount not valid if true.
  bool IsStreamed() const { return samp_count_ < 0; }

  /// Sample frequency, Hz.
  BaseFloat SampFreq() const { return samp_freq_; }

  /// Number of samples in stream. Invalid if IsStreamed() is true.
  uint32 SampleCount() const { return samp_count_; }

  /// Approximate duration, seconds. Invalid if IsStreamed() is true.
  BaseFloat Duration() const { return samp_count_ / samp_freq_; }

  /// Number of channels, 1 to 16.
  int32 NumChannels() const { return num_channels_; }

  /// Bytes per sample.
  size_t BlockAlign() const { return 2 * num_channels_; }

  /// Wave data bytes. Invalid if IsStreamed() is true.
  size_t DataBytes() const { return samp_count_ * BlockAlign(); }

  /// Is data file byte order different from machine byte order?
  bool ReverseBytes() const { return reverse_bytes_; }

  /// 'is' should be opened in binary mode. Read() will throw on error.
  /// On success 'is' will be positioned at the beginning of wave data.
  void Read(std::istream &is);

 private:
  BaseFloat samp_freq_;  //采样率
  int32 samp_count_;     // 0 if empty, -1 if undefined length.
  uint8 num_channels_;   //通道数
  bool reverse_bytes_;   // 文件时大端还是小端存储.
};
```

### WaveData

wav数据

```cpp
class WaveData {
 public:
  WaveData(BaseFloat samp_freq, const MatrixBase<BaseFloat> &data)
      : data_(data), samp_freq_(samp_freq) {}

  WaveData() : samp_freq_(0.0) {}

  /// Read() will throw on error.  It's valid to call Read() more than once--
  /// in this case it will destroy what was there before.
  /// "is" should be opened in binary mode.
  void Read(std::istream &is);

  /// Write() will throw on error.   os should be opened in binary mode.
  void Write(std::ostream &os) const;

  // This function returns the wave data-- it's in a matrix
  // because there may be multiple channels.  In the normal case
  // there's just one channel so Data() will have one row.
  const Matrix<BaseFloat> &Data() const { return data_; }

  BaseFloat SampFreq() const { return samp_freq_; }

  // Returns the duration in seconds
  BaseFloat Duration() const { return data_.NumCols() / samp_freq_; }

  void CopyFrom(const WaveData &other) {
    samp_freq_ = other.samp_freq_;
    data_.CopyFromMat(other.data_);
  }

  void Clear() {
    data_.Resize(0, 0);
    samp_freq_ = 0.0;
  }

  void Swap(WaveData *other) {
    data_.Swap(&(other->data_));
    std::swap(samp_freq_, other->samp_freq_);
  }

 private:
  static const uint32 kBlockSize = 1024 * 1024;  // Use 1M bytes.
  Matrix<BaseFloat> data_;  //数据，有可能是双通道的
  BaseFloat samp_freq_;  //采样率
};
```

### WaveHolder

wav文件的holder，只能读取wav文件，不能写wav文件

```cpp
class WaveHolder {
 public:
  typedef WaveData T;

  static bool Write(std::ostream &os, bool binary, const T &t) {
    // We don't write the binary-mode header here [always binary].
    if (!binary)
      KALDI_ERR << "Wave data can only be written in binary mode.";
    try {
      t.Write(os);  // throws exception on failure.
      return true;
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught in WaveHolder object (writing). "
                 << e.what();
      return false;  // write failure.
    }
  }
  void Copy(const T &t) { t_.CopyFrom(t); }

  static bool IsReadInBinary() { return true; }

  void Clear() { t_.Clear(); }

  T &Value() { return t_; }

  WaveHolder &operator = (const WaveHolder &other) {
    t_.CopyFrom(other.t_);
    return *this;
  }
  WaveHolder(const WaveHolder &other): t_(other.t_) {}

  WaveHolder() {}

  bool Read(std::istream &is) {
    // We don't look for the binary-mode header here [always binary]
    try {
      t_.Read(is);  // Throws exception on failure.
      return true;
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught in WaveHolder::Read(). " << e.what();
      return false;
    }
  }

  void Swap(WaveHolder *other) {
    t_.Swap(&(other->t_));
  }

  bool ExtractRange(const WaveHolder &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

 private:
  T t_;
};
```

### WaveInfoHolder

wav头的holder，读取wav头文件信息

```cpp
class WaveInfoHolder {
 public:
  typedef WaveInfo T;

  void Clear() { info_ = WaveInfo(); }
  void Swap(WaveInfoHolder *other) { std::swap(info_, other->info_); }
  T &Value() { return info_; }
  static bool IsReadInBinary() { return true; }

  bool Read(std::istream &is) {
    try {
      info_.Read(is);  // Throws exception on failure.
      return true;
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught in WaveInfoHolder::Read(). " << e.what();
      return false;
    }
  }

  bool ExtractRange(const WaveInfoHolder &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

 private:
  WaveInfo info_;
};
```



