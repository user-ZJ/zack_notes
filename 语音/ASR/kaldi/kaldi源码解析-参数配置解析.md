# kaldi源码解析-参数配置解析

kaldi中解析配置参数主要是src/util下面的parse-options.cc文件和parse-options.h。

配置参数解析的步骤为：

1. 注册配置参数
2. 解析配置参数，并对之前注册的配置参数进行匹配赋值

下面以计算mfcc的参数解析为例来详细说明配置参数解析的过程。

## 运行mfcc脚本，并提供相关参数

```shell
compute-mfcc-feats  --sample-frequency=16000 --use-energy=false scp:wav.scp ark:mfcc.ark
```

## compute-mfcc-feats中参数解析

compute-mfcc-feats源码在src/featbin/compute-mfcc-feats.cc中

仅查看参数解析部分代码：

```cpp
const char *usage =
        "Create MFCC feature files.\n"
        "Usage:  compute-mfcc-feats [options...] <wav-rspecifier> "
        "<feats-wspecifier>\n";

    // 1. 构造参数解析类的对象
    ParseOptions po(usage);
    MfccOptions mfcc_opts;
    bool subtract_mean = false;
    BaseFloat vtln_warp = 1.0;
    std::string vtln_map_rspecifier;
    std::string utt2spk_rspecifier;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;
    std::string output_format = "kaldi";
    std::string utt2dur_wspecifier;

    // 2. 将mfcc_opts中的配置注册到ParseOptions对象中
    mfcc_opts.Register(&po);

    // 3. 将全局参数注册到ParseOptions对象中
    po.Register("output-format", &output_format, "Format of the output "
                "files [kaldi, htk]");
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each "
                "feature file [CMS]; not recommended to do it this way. ");
    po.Register("vtln-warp", &vtln_warp, "Vtln warp factor (only applicable "
                "if vtln-map not specified)");
    po.Register("vtln-map", &vtln_map_rspecifier, "Map from utterance or "
                "speaker-id to vtln warp factor (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier, "Utterance to speaker-id map "
                "rspecifier (if doing VTLN and you have warps per speaker)");
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
                "0 -> left, 1 -> right)");
    po.Register("min-duration", &min_duration, "Minimum duration of segments "
                "to process (in seconds).");
    po.Register("write-utt2dur", &utt2dur_wspecifier, "Wspecifier to write "
                "duration of each utterance in seconds, e.g. 'ark,t:utt2dur'.");
    //4. ParseOptions对象读取命令行参数，并将值赋给之前注册的配置参数
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string wav_rspecifier = po.GetArg(1);
    std::string output_wspecifier = po.GetArg(2);
```

**注意：**po.Register传入的是注册参数的地址，在后面解析命令行参数的时候可以直接对注册的参数进行赋值

```cpp
po.Register("output-format", &output_format, "Format of the output "
                "files [kaldi, htk]");
```

## mfcc参数注册

调用MfccOptions的Register方法

```cpp
mfcc_opts.Register(&po);
```

实际上也是将mfcc中各个参数的地址注册到ParseOptions对象中

```cpp
// src/feat/feature-mfcc.h
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
```

## 将参数注册到ParseOptions对象中的过程

在ParseOptions类中重载了6个Register函数，只是传入的配置参数指针类型不同，分别有bool,int32,uint32,float,double,string

```cpp
// src/util/parse-options.h
void Register(const std::string &name,
                bool *ptr, const std::string &doc);
void Register(const std::string &name,
              int32 *ptr, const std::string &doc);
void Register(const std::string &name,
              uint32 *ptr, const std::string &doc);
void Register(const std::string &name,
              float *ptr, const std::string &doc);
void Register(const std::string &name,
              double *ptr, const std::string &doc);
void Register(const std::string &name,
              std::string *ptr, const std::string &doc);
```

以int类型为例，说明注册的过程

```cpp
// src/util/parse-options.cc
void ParseOptions::Register(const std::string &name,
                            int32 *ptr, const std::string &doc) {
  RegisterTmpl(name, ptr, doc);
}
```

**RegisterTmpl** 函数是一个模板函数，可以接收不同类型的配置参数

```cpp
// src/util/parse-options.cc
template<typename T>
void ParseOptions::RegisterTmpl(const std::string &name, T *ptr,
                                const std::string &doc) {
  if (other_parser_ == NULL) {  //other_parser_一般为NULL，使用kaldi的解析器
    this->RegisterCommon(name, ptr, doc, false);
  } else {
    KALDI_ASSERT(prefix_ != "" &&
                 "Cannot use empty prefix when registering with prefix.");
    std::string new_name = prefix_ + '.' + name;  // name becomes prefix.name
    other_parser_->Register(new_name, ptr, doc);
  }
}
```

**RegisterTmpl** 函数调用RegisterCommon同样是一个模板函数，接收不同类型的配置参数。其中的NormalizeArgName作用是将传出的字符串中的'_'转换为'-'，并将大写转换为小写。

```cpp
// src/util/parse-options.cc
template<typename T>
void ParseOptions::RegisterCommon(const std::string &name, T *ptr,
                                  const std::string &doc, bool is_standard) {
  KALDI_ASSERT(ptr != NULL);
  std::string idx = name;
  NormalizeArgName(&idx);
  //判断idx是否重复注册
  if (doc_map_.find(idx) != doc_map_.end())
    KALDI_WARN << "Registering option twice, ignoring second time: " << name;
  this->RegisterSpecific(name, idx, ptr, doc, is_standard);
}
```

**RegisterSpecific**将int指针保存在int_map_中，并将注册的其他信息保存在doc_map_中。

```cpp
// src/util/parse-options.cc
void ParseOptions::RegisterSpecific(const std::string &name,
                                    const std::string &idx,
                                    int32 *i,
                                    const std::string &doc,
                                    bool is_standard) {
  int_map_[idx] = i;
  std::ostringstream ss;
  ss << doc << " (int, default = " << *i << ")";
  doc_map_[idx] = DocInfo(name, ss.str(), is_standard);
}
```

ParseOptions对象中维护了以下的map ，用来存注册的各个类型的指针，一遍后续对注册的参数进行赋值

```cpp
// src/util/parse-options.h
std::map<std::string, bool*> bool_map_;
std::map<std::string, int32*> int_map_;
std::map<std::string, uint32*> uint_map_;
std::map<std::string, float*> float_map_;
std::map<std::string, double*> double_map_;
std::map<std::string, std::string*> string_map_;
```

## 解析参数，对注册参数进行赋值

```cpp
po.Read(argc, argv);
```

**Read**方法解析命令行参数，并对注册的参数进行赋值;其中SplitLongArg将--sample-frequency=16000形式分开为key-value的形式；

NormalizeArgName作用是将传出的字符串中的'_'转换为'-'，并将大写转换为小写；

如果有--config参数，则使用ReadConfigFile从配置文件中读取配置项

SetOption 将参数值赋值给注册的参数

```cpp
// src/util/parse-options.cc
int ParseOptions::Read(int argc, const char *const argv[]) {
  KALDI_LOG<<"parse read ";
  argc_ = argc;
  argv_ = argv;
  std::string key, value;
  int i;
  if (argc > 0) {
    // set global "const char*" g_program_name (name of the program)
    // so it can be printed out in error messages;
    // it's useful because often the stderr of different programs will
    // be mixed together in the same log file.
#ifdef _MSC_VER
    const char *c = strrchr(argv[0], '\\');
#else
    const char *c = strrchr(argv[0], '/');
#endif
    SetProgramName(c == NULL ? argv[0] : c + 1);
  }
  // 1. 查看命令行中是否存在--config或--help
  //如果存在--config 则读取config配置文件，并赋值给注册的配置
  //如果有--help，则打印Usage并退出
  for (i = 1; i < argc; i++) {
    if (std::strncmp(argv[i], "--", 2) == 0) {
      if (std::strcmp(argv[i], "--") == 0) {
        // a lone "--" marks the end of named options
        break;
      }
      bool has_equal_sign;
      // SplitLongArg将--sample-frequency=16000形式分开为key-value的形式
      SplitLongArg(argv[i], &key, &value, &has_equal_sign);
      NormalizeArgName(&key);
      Trim(&value);
      if (key.compare("config") == 0) {
        ReadConfigFile(value);
      }
      if (key.compare("help") == 0) {
        PrintUsage();
        exit(0);
      }
    }
  }
  bool double_dash_seen = false;
  // 2. 解析--sample-frequency=16000类型的参数
  for (i = 1; i < argc; i++) {
    if (std::strncmp(argv[i], "--", 2) == 0) {
      if (std::strcmp(argv[i], "--") == 0) {
        // A lone "--" marks the end of named options.
        // Skip that option and break the processing of named options
        i += 1;
        double_dash_seen = true;
        break;
      }
      bool has_equal_sign;
      SplitLongArg(argv[i], &key, &value, &has_equal_sign);
      NormalizeArgName(&key);
      Trim(&value);
      // 3. 将参数值赋值给注册的参数
      if (!SetOption(key, value, has_equal_sign)) {
        PrintUsage(true);
        KALDI_ERR << "Invalid option " << argv[i];
      }
    } else {
      break;
    }
  }
  // 4. 保存不带--的参数到positional_args_，通常是scp:xxx.scp和ark:xxx.ark
  for (; i < argc; i++) {
    if ((std::strcmp(argv[i], "--") == 0) && !double_dash_seen) {
      double_dash_seen = true;
    } else {
      positional_args_.push_back(std::string(argv[i]));
    }
  }
  // 5. 是否打印出所有命令行参数
  if (print_args_) {
    std::ostringstream strm;
    for (int j = 0; j < argc; j++)
      strm << Escape(argv[j]) << " ";
    strm << '\n';
    std::cerr << strm.str() << std::flush;
  }
  return i;
}
```

**ReadConfigFile**从配置文件中读取配置项，每一行是一条形如“--sample-frequency=16000”的配置，如果以'#'开头则跳过该行

SetOption 将参数值赋值给注册的参数

```cpp
void ParseOptions::ReadConfigFile(const std::string &filename) {
  std::ifstream is(filename.c_str(), std::ifstream::in);
  if (!is.good()) {
    KALDI_ERR << "Cannot open config file: " << filename;
  }

  std::string line, key, value;
  int32 line_number = 0;
  while (std::getline(is, line)) {
    line_number++;
    // trim out the comments
    size_t pos;
    if ((pos = line.find_first_of('#')) != std::string::npos) {
      line.erase(pos);
    }
    // skip empty lines
    Trim(&line);
    if (line.length() == 0) continue;

    if (line.substr(0, 2) != "--") {
      KALDI_ERR << "Reading config file " << filename
                << ": line " << line_number << " does not look like a line "
                << "from a Kaldi command-line program's config file: should "
                << "be of the form --x=y.  Note: config files intended to "
                << "be sourced by shell scripts lack the '--'.";
    }

    // parse option
    bool has_equal_sign;
    SplitLongArg(line, &key, &value, &has_equal_sign);
    NormalizeArgName(&key);
    Trim(&value);
    if (!SetOption(key, value, has_equal_sign)) {
      PrintUsage(true);
      KALDI_ERR << "Invalid option " << line << " in config file " << filename;
    }
  }
}
```

**SetOption** 将参数值赋值给注册的参数，使用key和注册的name进行匹配，将value赋值给匹配到的值。

```cpp
bool ParseOptions::SetOption(const std::string &key,
                             const std::string &value,
                             bool has_equal_sign) {
  if (bool_map_.end() != bool_map_.find(key)) {
    if (has_equal_sign && value == "")
      KALDI_ERR << "Invalid option --" << key << "=";
    *(bool_map_[key]) = ToBool(value);
  } else if (int_map_.end() != int_map_.find(key)) {
    *(int_map_[key]) = ToInt(value);
  } else if (uint_map_.end() != uint_map_.find(key)) {
    *(uint_map_[key]) = ToUint(value);
  } else if (float_map_.end() != float_map_.find(key)) {
    *(float_map_[key]) = ToFloat(value);
  } else if (double_map_.end() != double_map_.find(key)) {
    *(double_map_[key]) = ToDouble(value);
  } else if (string_map_.end() != string_map_.find(key)) {
    if (!has_equal_sign)
      KALDI_ERR << "Invalid option --" << key
                << " (option format is --x=y).";
    *(string_map_[key]) = value;
  } else {
    return false;
  }
  return true;
}
```

通过以上代码可以看出，如果--config文件中和命令行中的配置冲突，则使用命令行中的配置。如config文件中存在--sample-frequency=8000，同时命令行中存在--sample-frequency=16000，则最后使用的配置为--sample-frequency=16000。



