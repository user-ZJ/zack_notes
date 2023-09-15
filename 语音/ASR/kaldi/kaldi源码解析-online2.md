# kaldi源码解析-online2

kaldi中online的方式处理流式语言，即通过麦克风等设备不断获取语音并实时处理的场景。

以下通过kaldi源码来分析online的具体实现过程

## kaldi中online实例运行

```shell
online2-wav-dump-features ark:spk2utt scp:wav.scp ark:feat.ark
```

输入为spk2utt和wav.scp，内容为：

```text
#wav.scp
test_id 16k_5s.wav

#spk2utt
test test_id
```

其中16k_5s.wav为任意一条16k语音

## 源码解析

主要代码：

src/online2bin/online2-wav-dump-features.cc

src/online2/online-nnet2-feature-pipeline.cc

src/feat/online-feature.cc

### 主函数

首先看一下主函数，省去一些不必要的代码如下，具体解释我们在代码中以注释方式体现：

```cpp
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    //特征提取配置项，后面详细展开
    OnlineNnet2FeaturePipelineConfig feature_config;
    BaseFloat chunk_length_secs = 0.05;
    bool print_ivector_dim = false;
    OnlineNnet2FeaturePipelineInfo feature_info(feature_config);

    std::string spk2utt_rspecifier = po.GetArg(1),  //ark:spk2utt
        wav_rspecifier = po.GetArg(2),     //scp:wav.scp
        feats_wspecifier = po.GetArg(3);   //ark:feat.ark

    int32 num_done = 0, num_err = 0;
    int64 num_frames_tot = 0;
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    BaseFloatMatrixWriter feats_writer(feats_wspecifier);
    //读取spk2utt 示例中只有一行，为"test test_id"
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();  //test
      const std::vector<std::string> &uttlist = spk2utt_reader.Value(); //test_id
      //使用的是默认配置，只提取mfcc特征
      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];  //test_id
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        //获取wav数据，默认获取第一个信道的数据
        const WaveData &wave_data = wav_reader.Value(utt);
        SubVector<BaseFloat> data(wave_data.Data(), 0);
        //使用的是默认配置，只提取mfcc特征
        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);
        //存放提取的特征
        std::vector<Vector<BaseFloat> *> feature_data;

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length = int32(samp_freq * chunk_length_secs);
        if (chunk_length == 0) chunk_length = 1;

        int32 samp_offset = 0;
        while (samp_offset < data.Dim()) {
          //每次取num_samp音频数据，直到取完所有数据
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;
          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          //将一段语音送入feature_pipeline进行特征提取
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);
          samp_offset += num_samp;
          //如果整个语音处理完成，调用InputFinished函数结束处理，
          //InputFinished会把buffer中的所有语音提取特征,保证结束的时候所有语音都提取了特征
          if (samp_offset == data.Dim())  // no more input. flush out last frames
            feature_pipeline.InputFinished();
          //如果feature_data中的数据比已经提取的特征少，则将已经提取的特征拷贝到feature_data
          while (static_cast<int32>(feature_data.size()) <
                 feature_pipeline.NumFramesReady()) {
            int32 t = static_cast<int32>(feature_data.size());
            feature_data.push_back(new Vector<BaseFloat>(feature_pipeline.Dim(),
                                                         kUndefined));
            //拷贝特征
            feature_pipeline.GetFrame(t, feature_data.back());
          }
        }
        int32 T = static_cast<int32>(feature_data.size());
        if (T == 0) {
          KALDI_WARN << "Got no frames of data for utterance " << utt;
          num_err++;
          continue;
        }
        //将feature_pipeline数据规整到Matrix
        Matrix<BaseFloat> feats(T, feature_pipeline.Dim());
        for (int32 t = 0; t < T; t++) {
          feats.Row(t).CopyFromVec(*(feature_data[t]));
          delete feature_data[t];
        }
        num_frames_tot += T;
        feats_writer.Write(utt, feats);
        feature_pipeline.GetAdaptationState(&adaptation_state);
        num_done++;
      }
    }
    KALDI_LOG << "Processed " << num_done << " utterances, "
              << num_err << " with errors; " << num_frames_tot
              << " frames in total.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
```

### 配置说明

#### OnlineNnet2FeaturePipelineConfig

OnlineNnet2FeaturePipelineConfig特征提取的配置，因为没有传入配置，所有都是默认的配置，这里使用的是构造函数的默认参数

feature_type("mfcc"), add_pitch(false)

```cpp
struct OnlineNnet2FeaturePipelineConfig {
  std::string feature_type;  // "plp" or "mfcc" or "fbank"
  std::string mfcc_config;
  std::string plp_config;
  std::string fbank_config;
  std::string cmvn_config;
  std::string global_cmvn_stats_rxfilename;
  bool add_pitch;
  std::string online_pitch_config;
  std::string ivector_extraction_config;
  OnlineSilenceWeightingConfig silence_weighting_config;

  OnlineNnet2FeaturePipelineConfig():
      feature_type("mfcc"), add_pitch(false) { }
};
```

#### OnlineNnet2FeaturePipelineInfo

解析OnlineNnet2FeaturePipelineConfig中的配置，如果没有提供配置，则使用默认配置。

OnlineNnet2FeaturePipelineConfig中默认add_pitch为false，所以OnlineNnet2FeaturePipelineInfo中add_pitch为false

cmvn_config和ivector_extraction_config为空，所以use_cmvn和use_ivectors为false

在默认情况下只是提取mfcc特征。

```cpp
struct OnlineNnet2FeaturePipelineInfo {
  OnlineNnet2FeaturePipelineInfo():
      feature_type("mfcc"), add_pitch(false), use_cmvn(false) { }
  OnlineNnet2FeaturePipelineInfo(const OnlineNnet2FeaturePipelineConfig &config);
  BaseFloat FrameShiftInSeconds() const;
  std::string feature_type; /// "mfcc" or "plp" or "fbank"
  MfccOptions mfcc_opts;  /// options for MFCC computation,
                          /// if feature_type == "mfcc"
  PlpOptions plp_opts;    /// Options for PLP computation, if feature_type == "plp"
  FbankOptions fbank_opts;  /// Options for filterbank computation, if
                            /// feature_type == "fbank"
  bool add_pitch;
  PitchExtractionOptions pitch_opts;  /// Options for pitch extraction, if done.
  ProcessPitchOptions pitch_process_opts;  /// Options for pitch post-processing
  bool use_cmvn;
  OnlineCmvnOptions cmvn_opts; /// Options for online cmvn, read from config file.
  std::string global_cmvn_stats_rxfilename;  /// Filename used for reading global
  bool use_ivectors;
  OnlineIvectorExtractionInfo ivector_extractor_info;
  OnlineSilenceWeightingConfig silence_weighting_config;
  BaseFloat GetSamplingFrequency();
  int32 IvectorDim() { return ivector_extractor_info.extractor.IvectorDim(); }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineNnet2FeaturePipelineInfo);
};
```

### OnlineNnet2FeaturePipeline（重要）

OnlineNnet2FeaturePipeline是处理online语音的pipeline，主要是接收语音，将接收的语音存入到内部buffer，从buffer中取语音进行特征提取，并将提取的特征拷贝目标位置，主要有3个函数完成

**AcceptWaveform**：接收语音，存入内部buffer

**ComputeFeatures**：计算特征

**GetFrame**：将特征拷贝到指定位置

```cpp
class OnlineNnet2FeaturePipeline: public OnlineFeatureInterface {
 public:
  explicit OnlineNnet2FeaturePipeline(
      const OnlineNnet2FeaturePipelineInfo &info);

  /// Member functions from OnlineFeatureInterface:
  virtual int32 Dim() const;

  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  void UpdateFrameWeights(
      const std::vector<std::pair<int32, BaseFloat> > &delta_weights);

  void SetAdaptationState(
      const OnlineIvectorExtractorAdaptationState &adaptation_state);

  void GetAdaptationState(
      OnlineIvectorExtractorAdaptationState *adaptation_state) const;

  void SetCmvnState(const OnlineCmvnState &cmvn_state);
  void GetCmvnState(OnlineCmvnState *cmvn_state);

  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);

  BaseFloat FrameShiftInSeconds() const { return info_.FrameShiftInSeconds(); }
  void InputFinished();

  OnlineIvectorFeature *IvectorFeature() {
    return ivector_feature_;
  }

  const OnlineIvectorFeature *IvectorFeature() const {
    return ivector_feature_;
  }

  OnlineFeatureInterface *InputFeature() {
    return nnet3_feature_;
  }

  virtual ~OnlineNnet2FeaturePipeline();

 private:
  const OnlineNnet2FeaturePipelineInfo &info_;
  OnlineBaseFeature *base_feature_;    /// MFCC/PLP/filterbank
  OnlinePitchFeature *pitch_;          /// Raw pitch, if used
  OnlineProcessPitch *pitch_feature_;  /// Processed pitch, if pitch used.
  OnlineCmvn *cmvn_feature_;
  Matrix<BaseFloat> lda_mat_;          /// LDA matrix, if supplied
  Matrix<double> global_cmvn_stats_;   /// Global CMVN stats.
  OnlineFeatureInterface *feature_plus_optional_pitch_;
  OnlineFeatureInterface *feature_plus_optional_cmvn_;
  OnlineIvectorFeature *ivector_feature_;  /// iVector feature, if used.
  OnlineFeatureInterface *nnet3_feature_;
  OnlineFeatureInterface *final_feature_;
  int32 dim_;
};
```

OnlineNnet2FeaturePipelineInfo的默认配置中，feature_type == "mfcc"，add_pitch=false，use_cmvn=false，use_ivectors=false；所以默认情况下只提取OnlineMfcc特征。

```cpp
OnlineNnet2FeaturePipeline::OnlineNnet2FeaturePipeline(
    const OnlineNnet2FeaturePipelineInfo &info):
    info_(info), base_feature_(NULL),
    pitch_(NULL), pitch_feature_(NULL),
    cmvn_feature_(NULL),
    feature_plus_optional_pitch_(NULL),
    feature_plus_optional_cmvn_(NULL),
    ivector_feature_(NULL),
    nnet3_feature_(NULL),
    final_feature_(NULL) {

  if (info_.feature_type == "mfcc") {
    base_feature_ = new OnlineMfcc(info_.mfcc_opts);
  } else if (info_.feature_type == "plp") {
    base_feature_ = new OnlinePlp(info_.plp_opts);
  } else if (info_.feature_type == "fbank") {
    base_feature_ = new OnlineFbank(info_.fbank_opts);
  } else {
    KALDI_ERR << "Code error: invalid feature type " << info_.feature_type;
  }

  if (info_.add_pitch) {
    pitch_ = new OnlinePitchFeature(info_.pitch_opts);
    pitch_feature_ = new OnlineProcessPitch(info_.pitch_process_opts,
                                            pitch_);
    feature_plus_optional_pitch_ = new OnlineAppendFeature(base_feature_,
                                                           pitch_feature_);
  } else {
    feature_plus_optional_pitch_ = base_feature_;
  }
  if (info_.use_cmvn) {
    KALDI_ASSERT(info.global_cmvn_stats_rxfilename != "");
    ReadKaldiObject(info.global_cmvn_stats_rxfilename, &global_cmvn_stats_);
    OnlineCmvnState initial_state(global_cmvn_stats_);
    cmvn_feature_ = new OnlineCmvn(info_.cmvn_opts, initial_state,
        feature_plus_optional_pitch_);
    feature_plus_optional_cmvn_ = cmvn_feature_;
  } else {
    feature_plus_optional_cmvn_ = feature_plus_optional_pitch_;
  }

  if (info_.use_ivectors) {
    nnet3_feature_ = feature_plus_optional_cmvn_;
    // Note: the i-vector extractor OnlineIvectorFeature gets 'base_feautre_'
    // without cmvn (the online cmvn is applied inside the class)
    ivector_feature_ = new OnlineIvectorFeature(info_.ivector_extractor_info,
                                                base_feature_);
    final_feature_ = new OnlineAppendFeature(feature_plus_optional_cmvn_,
                                             ivector_feature_);
  } else {
    nnet3_feature_ = feature_plus_optional_cmvn_;
    final_feature_ = feature_plus_optional_cmvn_;
  }
  dim_ = final_feature_->Dim();
}
```

#### AcceptWaveform

在main函数中调用feature_pipeline.AcceptWaveform(samp_freq, wave_part);函数如下，会调用base_feature_的AcceptWaveform函数；base_feature_在初始化的时候被初始化为OnlineMfcc，所以调用的是OnlineMfcc的AcceptWaveform

```cpp
void OnlineNnet2FeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
  if (pitch_)
    pitch_->AcceptWaveform(sampling_rate, waveform);
}
```

因为OnlineMfcc实际上是OnlineGenericBaseFeature类型，所用调用的是OnlineGenericBaseFeature<C>::AcceptWaveform函数

```cpp
typedef OnlineGenericBaseFeature<MfccComputer> OnlineMfcc;
```

OnlineGenericBaseFeature<C>::AcceptWaveform函数中将传入的一段语音追加到waveform_remainder_，并调用ComputeFeatures()计算mfcc特征。

```cpp
template <class C>
void OnlineGenericBaseFeature<C>::AcceptWaveform(
    BaseFloat sampling_rate, const VectorBase<BaseFloat> &original_waveform) {
  if (original_waveform.Dim() == 0)
    return;  // Nothing to do.
  if (input_finished_)
    KALDI_ERR << "AcceptWaveform called after InputFinished() was called.";

  Vector<BaseFloat> appended_wave;
  Vector<BaseFloat> resampled_wave;

  const VectorBase<BaseFloat> *waveform;

  MaybeCreateResampler(sampling_rate);
  if (resampler_ == nullptr) {
    waveform = &original_waveform;
  } else {
    resampler_->Resample(original_waveform, false, &resampled_wave);
    waveform = &resampled_wave;
  }

  appended_wave.Resize(waveform_remainder_.Dim() + waveform->Dim());
  if (waveform_remainder_.Dim() != 0)
    appended_wave.Range(0, waveform_remainder_.Dim())
        .CopyFromVec(waveform_remainder_);
  appended_wave.Range(waveform_remainder_.Dim(), waveform->Dim())
      .CopyFromVec(*waveform);
  waveform_remainder_.Swap(&appended_wave);
  ComputeFeatures();
}
```

#### ComputeFeatures

ComputeFeatures从waveform_remainder_中取一段语音，提取特征后将特征放到features_中

```cpp
template <class C>
void OnlineGenericBaseFeature<C>::ComputeFeatures() {
  const FrameExtractionOptions &frame_opts = computer_.GetFrameOptions();
  int64 num_samples_total = waveform_offset_ + waveform_remainder_.Dim();
  int32 num_frames_old = features_.Size(),
      num_frames_new = NumFrames(num_samples_total, frame_opts,
                                 input_finished_);
  KALDI_ASSERT(num_frames_new >= num_frames_old);

  Vector<BaseFloat> window;
  bool need_raw_log_energy = computer_.NeedRawLogEnergy();
  //提取waveform_remainder_中num_frames_old到num_frames_new长度的mfcc特征
  //计算出来的特征存放到features_中
  for (int32 frame = num_frames_old; frame < num_frames_new; frame++) {
    BaseFloat raw_log_energy = 0.0;
    ExtractWindow(waveform_offset_, waveform_remainder_, frame,
                  frame_opts, window_function_, &window,
                  need_raw_log_energy ? &raw_log_energy : NULL);
    Vector<BaseFloat> *this_feature = new Vector<BaseFloat>(computer_.Dim(),
                                                            kUndefined);
    // note: this online feature-extraction code does not support VTLN.
    BaseFloat vtln_warp = 1.0;
    computer_.Compute(raw_log_energy, vtln_warp, &window, this_feature);
    features_.PushBack(this_feature);  //features_为RecyclingVector
  }
  int64 first_sample_of_next_frame = FirstSampleOfFrame(num_frames_new,
                                                        frame_opts);
  int32 samples_to_discard = first_sample_of_next_frame - waveform_offset_;
  if (samples_to_discard > 0) {
    int32 new_num_samples = waveform_remainder_.Dim() - samples_to_discard;
    if (new_num_samples <= 0) {
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
```

#### GetFrame

将指定帧的特征拷贝到feat中

```cpp
template <class C>
void OnlineGenericBaseFeature<C>::GetFrame(int32 frame,
                                           VectorBase<BaseFloat> *feat) {
  feat->CopyFromVec(*(features_.At(frame)));
};
```

## 总结

在online2-wav-dump-features中演示了online提取语音特征的流程，接收语音->将语音存到buffer中->从buffer中取语音提取特征->将特征返回。主要思路是利用buffer，将语音分成多段处理，从而达到可以源源不断处理语音的效果。

在本样例中使用的是单个线程，在使用中可以使用多个线程协同，利用生产者-消费者模型进行特征提取。