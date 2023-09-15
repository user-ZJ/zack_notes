# kaldi源码解析-计算mfcc

## 运行

kaldi计算mfcc代码位于src/featbin/compute-mfcc-feats.cc；编译后的bin名称为compute-mfcc-feats

准备wav.scp文件，以便运行compute-mfcc-feats，内容如下：

```
# id wav_path
test /path/to/test.wav
```

使用如下命令提取mfcc特征

```shell
# 使用--allow-upsample=true --allow-downsample=true参数可以提取任何采样率的wav的mfcc特征
# ark,t:mfcc.ark使得输出是文本格式，方便查看
compute-mfcc-feats --allow-upsample=true --allow-downsample=true scp:wav.scp ark,t:mfcc.ark
```

## 代码解析

提取compute-mfcc-feats.cc主干代码如下：

```cpp
int main(int argc, char *argv[]) {
  try {
    // scp:wav.scp
    std::string wav_rspecifier = po.GetArg(1);
    // ark,t:mfcc.ark
    std::string output_wspecifier = po.GetArg(2);

    Mfcc mfcc(mfcc_opts);
    // 顺序从wav.scp中读取每行数据
    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key(); // 读取utt,在本例中对应于test
      const WaveData &wave_data = reader.Value(); //读取语音，加载/path/to/test.wav
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        continue;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      BaseFloat vtln_warp_local;  // Work out VTLN warp factor.
      {
        vtln_warp_local = vtln_warp;
      }

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        // 核心调用，提取MFCC特征
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(),
                             vtln_warp_local, &features);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance " << utt;
        continue;
      }
      if (output_format == "kaldi") {
        kaldi_writer.Write(utt, features);
      } 
      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
```

Mfcc定义在src/feat/feature-mfcc.h中

```cpp
typedef OfflineFeatureTpl<MfccComputer> Mfcc;
```

OfflineFeatureTpl定义在src/feat/feature-common.h中，在OfflineFeatureTpl构造函数中会创建一个MfccComputer类型的compute_

```cpp
OfflineFeatureTpl(const Options &opts):
      computer_(opts),
      feature_window_function_(computer_.GetFrameOptions()) { }
```

mfcc.ComputeFeatures实现在src/feat/feature-common-inl.h中，首先会判断音频采样率是否相同，不同的话会对音频进行重采样，然后调用OfflineFeatureTpl::Compute()

```cpp
template <class F>
void OfflineFeatureTpl<F>::ComputeFeatures(
    const VectorBase<BaseFloat> &wave,
    BaseFloat sample_freq,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  BaseFloat new_sample_freq = computer_.GetFrameOptions().samp_freq;
  if (sample_freq == new_sample_freq) {
    Compute(wave, vtln_warp, output);
  } else {
    if (new_sample_freq < sample_freq &&
        ! computer_.GetFrameOptions().allow_downsample)
        KALDI_ERR << "Waveform and config sample Frequency mismatch: "
                  << sample_freq << " .vs " << new_sample_freq
                  << " (use --allow-downsample=true to allow "
                  << " downsampling the waveform).";
    else if (new_sample_freq > sample_freq &&
             ! computer_.GetFrameOptions().allow_upsample)
      KALDI_ERR << "Waveform and config sample Frequency mismatch: "
                  << sample_freq << " .vs " << new_sample_freq
                << " (use --allow-upsample=true option to allow "
                << " upsampling the waveform).";
    // Resample the waveform.
    Vector<BaseFloat> resampled_wave(wave);
    ResampleWaveform(sample_freq, wave,
                     new_sample_freq, &resampled_wave);
    Compute(resampled_wave, vtln_warp, output);
  }
}
```

OfflineFeatureTpl::Compute()回对音频进行分帧，加窗的操作；然后再调用computer_.Compute计算mfcc

```cpp
template <class F>
void OfflineFeatureTpl<F>::Compute(
    const VectorBase<BaseFloat> &wave,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  int32 rows_out = NumFrames(wave.Dim(), computer_.GetFrameOptions()),
      cols_out = computer_.Dim();
  if (rows_out == 0) {
    output->Resize(0, 0);
    return;
  }
  output->Resize(rows_out, cols_out);
  Vector<BaseFloat> window;  // windowed waveform.
  bool use_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    BaseFloat raw_log_energy = 0.0;
    // 分帧，加窗
    ExtractWindow(0, wave, r, computer_.GetFrameOptions(),
                  feature_window_function_, &window,
                  (use_raw_log_energy ? &raw_log_energy : NULL));

    SubVector<BaseFloat> output_row(*output, r);
    computer_.Compute(raw_log_energy, vtln_warp, &window, &output_row);
  }
}
```

computer_.Compute实现在src/feat/feature-mfcc.cc中的MfccComputer::Compute，对每一帧计算fft，梅尔滤波，对数能量，DCT等操作

```cpp
void MfccComputer::Compute(BaseFloat signal_raw_log_energy,
                           BaseFloat vtln_warp,
                           VectorBase<BaseFloat> *signal_frame,
                           VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());

  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  if (opts_.use_energy && !opts_.raw_energy)
    signal_raw_log_energy = Log(std::max<BaseFloat>(VecVec(*signal_frame, *signal_frame),
                                     std::numeric_limits<float>::epsilon()));
  // 计算fft
  if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two.
    RealFft(signal_frame, true);

  ComputePowerSpectrum(signal_frame);
  SubVector<BaseFloat> power_spectrum(*signal_frame, 0,
                                      signal_frame->Dim() / 2 + 1);
  // 梅尔滤波
  mel_banks.Compute(power_spectrum, &mel_energies_);

  // 计算对数能量
  mel_energies_.ApplyFloor(std::numeric_limits<float>::epsilon());
  mel_energies_.ApplyLog();  // take the log.

  feature->SetZero();  // in case there were NaNs.
  // 计算DCT
  // feature = dct_matrix_ * mel_energies [which now have log]
  feature->AddMatVec(1.0, dct_matrix_, kNoTrans, mel_energies_, 0.0);

  if (opts_.cepstral_lifter != 0.0)
    feature->MulElements(lifter_coeffs_);

  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0 && signal_raw_log_energy < log_energy_floor_)
      signal_raw_log_energy = log_energy_floor_;
    (*feature)(0) = signal_raw_log_energy;
  }
}
```



