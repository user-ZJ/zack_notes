# kaldi源码解析-cudadecoder

lanes:相当于神经网络中批次大小，表示一组正在被解码的语音或流

channels：未准备好继续处理的语音的保持状态，即正在准备的batch数据

```cpp
typedef float CostType;
typedef int32 StateId;
typedef int32 IntegerCostType;
typedef int32 LaneId;
typedef int32 ChannelId;
// int2为cuda中数据类型，表示长度为2的int vector，通过.x和.y访问元素
typedef int32 LatticeStateInternalId;
typedef StateId OutputLatticeState;
typedef int32 TokenId;

typedef fst::VectorFst<LatticeArc> Lattice;
```





## CudaFst

gpu解码中fst表示形式，同时存储在host和device设备上

CudaFst格式是将OpenFst格式存储为Compressed Sparse Row ([CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))) Matrix（压缩稀疏行 (CSR) 矩阵）格式

 states = rows   arcs = columns.

CSR格式使FST 存储更紧凑,访问内存更方便

例如，当从给定源加载arc时，我们可以通过合并读取加载所有arc信息（destination, weight等） 发射arcs和非发射arcs作为单独的矩阵存储以提高效率

然后我们将 FST 复制到设备（同时在主机上保留其原始副本）

```cpp
class CudaFst {
 public:
  CudaFst()
      : d_e_offsets_(nullptr),
        d_ne_offsets_(nullptr),
        d_arc_weights_(nullptr),
        d_arc_nextstates_(nullptr),
        d_arc_pdf_ilabels_(nullptr),
        d_final_(nullptr){};
  // 创建FST的CSR表示,并拷贝到GPU
  // 如果TransitionModel不为空, 则使用TransitionModel将ilabels id indexes 转换为 pdf indexes
  // 如果TransitionModel为空, 则TransitionModel == identity
  // 重要: CudaDecodable 不使用TransitionModel. 如果使用了TransitionModel, 需要再此处传入
  void Initialize(const fst::Fst<StdArc> &fst,
                  const TransitionModel *trans_model = NULL);
  void Finalize();  //释放内存
  // 获取状态数
  inline uint32_t NumStates() const { return num_states_; }
  inline StateId Start() const { return start_; }  //获取起始状态

 private:
  friend class CudaDecoder;
  // 计算状态总数；计算Emitting/Non-emitting/All arcs总数；计算fst的offsets
  void ComputeOffsets(const fst::Fst<StdArc> &fst);
  // Allocates memory to store FST
  void AllocateData(const fst::Fst<StdArc> &fst);
  // 填充arc数据（h_arc_weights_、h_arc_nextstate_ 等）
  void PopulateArcs(const fst::Fst<StdArc> &fst);
  // 使用TransitionModel模型将 id ilabels 转换为 pdf ilabels
  // 它允许 CudaDecoder 读取正确索引处的声学模型对数似然
  void ApplyTransitionModelOnIlabels(const TransitionModel &trans_model);
  // Copies fst to device into the pre-allocated datastructures
  void CopyDataToDevice();
  // FST中状态总数
  unsigned int num_states_;
  //  FST的起始状态，解码需要从起始状态开始
  StateId start_;
  // Number of emitting, non-emitting, and total number of arcs
  unsigned int e_count_, ne_count_, arc_count_;
  // Offset arrays长度为num_states_+1
  // 状态 i 的Arc values存储在 [offset[i],offset[i+1]]
  unsigned int *d_e_offsets_;  // Emitting offset arrays
  std::vector<unsigned int> h_e_offsets_;
  unsigned int *d_ne_offsets_;  // Non-emitting offset arrays
  std::vector<unsigned int> h_ne_offsets_;
  // 状态i的Arcs 存储在 [offsets[i],offsets[i+1]]
  // 使用e_offsets查找emitting arc，使用ne_offsets查找nonemitting arc
  // ilabels数组长度为e_count_，而不是arc_count_
  std::vector<CostType> h_arc_weights_;  // arc上的权重数组
  CostType *d_arc_weights_;
  std::vector<StateId> h_arc_nextstate_;  // arc下一跳状态数组
  StateId *d_arc_nextstates_;
  std::vector<int32> h_arc_id_ilabels_;  //输入标签id数组
  int32 *d_arc_pdf_ilabels_;
  std::vector<int32> h_arc_olabels_;   //输出标签id数组
  // Final costs；h_final_[i]表示状态i的final cost
  std::vector<CostType> h_final_;
  CostType *d_final_;

  // ilabels (pdf indexing)
  // only populate during CSR generation, cleared after (not needed on host)
  std::vector<int32> h_arc_pdf_ilabels_;
};
```







## CudaDecodableInterface

CudaDecodableInterface在DecodableInterface接口上新增了GetLogLikelihoodsCudaPointer接口

```cpp
class CudaDecodableInterface : public DecodableInterface {
public:
  virtual BaseFloat *GetLogLikelihoodsCudaPointer(int32 subsampled_frame) = 0;
};
```

## DecodableCuMatrixMapped

DecodableCuMatrixMapped实现了CudaDecodableInterface接口

```cpp
class DecodableCuMatrixMapped : public CudaDecodableInterface {
public:
  // This constructor creates an object that will not delete "likes" when done.
  // the frame_offset is the frame the row 0 of 'likes' corresponds to, would be
  // greater than one if this is not the first chunk of likelihoods.
  DecodableCuMatrixMapped(const TransitionModel &tm,
                          const CuMatrixBase<BaseFloat> &likes,
                          int32 frame_offset = 0);

  virtual int32 NumFramesReady() const;

  virtual bool IsLastFrame(int32 frame) const;

  //空实现，不会被使用到
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    KALDI_ASSERT(false);
    return 0.0f;  // never executed, compiler requests a return
  };

  // Note: these indices are 1-based.
  virtual int32 NumIndices() const;

  virtual ~DecodableCuMatrixMapped(){};

  // 返回声学模型打分的GPU内存指针,可以使用CuValue<float>打印
  virtual BaseFloat *GetLogLikelihoodsCudaPointer(int32 subsampled_frame);

private:
  const TransitionModel &trans_model_; // for tid to pdf mapping
  const CuMatrixBase<BaseFloat> *likes_;

  int32 frame_offset_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableCuMatrixMapped);
};
```



## CudaDecoderConfig

cuda解码的配置项

```cpp
struct CudaDecoderConfig {
  BaseFloat default_beam;
  BaseFloat lattice_beam;
  int32 ntokens_pre_allocated;
  int32 main_q_capacity, aux_q_capacity;
  int32 max_active;
  OnlineEndpointConfig endpointing_config;

  CudaDecoderConfig()
      : default_beam(15.0),  //beam参数,剪枝强度
        lattice_beam(10.0),  //lattice-beam参数，剪枝强度
        ntokens_pre_allocated(1000000), //
        main_q_capacity(-1),
        aux_q_capacity(-1),
        max_active(10000) {}
};
```

##  CudaDecoder

cuda解码的实现

### 数据结构

```cpp
typedef StateId OutputLatticeState;
struct RawLatticeState {
    CostType token_extra_cost; //从当前lattice_state 到最终帧的所有路径的extra_cost 的最小值。
    OutputLatticeState fst_lattice_state; //fst_out 中lattice_state 的StateId
    bool is_state_closed; // 如果token_extra_cost 已被另一个token读取
  };
```



```cpp
// InfoToken contains data that needs to be saved for the backtrack
// in GetBestPath/GetRawLattice
// We don't need the token.cost or token.next_state.
struct __align__(8) InfoToken {
  int32 prev_token;  //如果(frame,fst_state)有多个token时使用，prev_token表示在extra_prev_tokens中的偏移
  int32 arc_idx;  //arc索引，如果(frame,fst_state)有多个token时使用，-arc_idx表示在extra_prev_tokens中的token个数
  bool IsUniqueTokenForStateAndFrame() {
    // This is a trick used to save space and PCI-E bandwidth (cf preprocess_in_place kernel)
    // This token is associated with a next_state s, created during the processing of frame f.
    // 如果帧的state s关联到多个tokens, arc_idx < 0 ; -arc_idx 表示tokens数.
    // 然后我们将不得不查看另一个列表来读取实际的 arc_idx 和 prev_token 值
    // 如果当前token只有一个，prev_token和arc_idx是有效的，可以直接使用
    return (arc_idx >= 0);
  }

  // 如果此令牌链接到同一帧中的其他令牌，则调用 (cf comments for IsUniqueTokenForStateAndFrame)
  // 返回 {offset,size} 对,在extra_prev_tokens列表中查找所有token
  // They are stored at offset "offset", and we have "size" of those
  std::pair<int32, int32> GetSameFSTStateTokensList() {
    KALDI_ASSERT(!IsUniqueTokenForStateAndFrame());

    return {prev_token, -arc_idx};
  }
};
```



```cpp
// Hashmap value. Used when computing the hashmap in PostProcessingMainQueue
struct __align__(16) HashmapValueT {
  // Map key : fst state
  int32 key;
  // Number of tokens associated to that state
  int32 count;
  // minimum cost for that state + argmin
  unsigned long long min_and_argmin_int_cost_u64;
};
```



```cpp
// 在设备上，我们批量计算所有内容
// 数据存储为 2D 矩阵（BatchSize、1D_Size）
// 例如，对于令牌队列，(BatchSize, max_tokens_per_frame_)
// DeviceMatrix 拥有数据但不用于访问它。
// 要实际访问数据，我们应该通过 GetView 请求视图
// 该视图包含用于访问数据的主机 cuda 代码。 它不拥有数据。
template <typename T>
// if necessary, make a version that always use ncols_ as the next power of 2
class DeviceMatrix {
  T *data_;
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ncols_ > 0);
    KALDI_ASSERT(!data_);
    data_ = static_cast<T *>(CuDevice::Instantiate().Malloc(
        (size_t)nrows_ * ncols_ * sizeof(*data_)));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    CuDevice::Instantiate().Free(data_);
  }

 protected:
  int32 ncols_;
  int32 nrows_;

 public:
  DeviceMatrix() : data_(NULL), ncols_(0), nrows_(0) {}

  virtual ~DeviceMatrix() {
    if (data_) Free();
  }

  void Resize(int32 nrows, int32 ncols) {
    if (data_) Free();
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ncols > 0);
    nrows_ = nrows;
    ncols_ = ncols;
    Allocate();
  }

  T *MutableData() {
    KALDI_ASSERT(data_);
    return data_;
  }
  // abstract getInterface...
};

template <typename T>
// if necessary, make a version that always use ncols_ as the next power of 2
class HostMatrix {
  T *data_;
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ncols_ > 0);
    KALDI_ASSERT(!data_);
    cudaMallocHost((void **)&data_, (size_t)nrows_ * ncols_ * sizeof(*data_));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    cudaFreeHost(data_);
  }

 protected:
  int32 ncols_;
  int32 nrows_;

 public:
  HostMatrix() : data_(NULL), ncols_(0), nrows_(0) {}

  virtual ~HostMatrix() {
    if (data_) Free();
  }

  void Resize(int32 nrows, int32 ncols) {
    if (data_) Free();
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ncols > 0);
    nrows_ = nrows;
    ncols_ = ncols;
    Allocate();
  }

  T *MutableData() {
    KALDI_ASSERT(data_);
    return data_;
  }
  // abstract getInterface...
};
// DeviceMatrix的视图，可以直接访问DeviceMatrix。可以通过复制View来达到传递DeviceMatrix的作用
template <typename T>
struct LaneMatrixView {
  T *data_;
  int32 ncols_;
  __host__ __device__ __inline__ T *lane(const int32 ilane) {
    return &data_[ilane * ncols_];
  }
};

template <typename T>
struct ChannelMatrixView {
  T *data_;
  int32 ncols_;
  __host__ __device__ __inline__ T *channel(const int32 ichannel) {
    return &data_[ichannel * ncols_];
  }
};

// lane 和 channel 的DeviceMatrix
template <typename T>
class DeviceLaneMatrix : public DeviceMatrix<T> {
 public:
  LaneMatrixView<T> GetView() { return {this->MutableData(), this->ncols_}; }

  T *lane(const int32 ilane) {
    return &this->MutableData()[ilane * this->ncols_];
  }
};

template <typename T>
class HostLaneMatrix : public HostMatrix<T> {
 public:
  LaneMatrixView<T> GetView() { return {this->MutableData(), this->ncols_}; }

  T *lane(const int32 ilane) {
    return &this->MutableData()[ilane * this->ncols_];
  }
};

template <typename T>
class DeviceChannelMatrix : public DeviceMatrix<T> {
 public:
  ChannelMatrixView<T> GetView() { return {this->MutableData(), this->ncols_}; }
  T *channel(const int32 ichannel) {
    return &this->MutableData()[ichannel * this->ncols_];
  }
};
```



```cpp
//DeviceParams 包含kernels使用的所有top-level常量数据，即在内核调用之间不会改变的数据（例如指向 main_q 的内存指针）
struct DeviceParams {
  ChannelMatrixView<ChannelCounters> d_channels_counters;
  LaneMatrixView<LaneCounters> d_lanes_counters;
  LaneMatrixView<LaneCounters> h_lanes_counters;

  ChannelMatrixView<int2> d_main_q_state_and_cost;
  ChannelMatrixView<int32> d_main_q_degrees_prefix_sum;
  ChannelMatrixView<int32> d_main_q_arc_offsets;
  LaneMatrixView<CostType> d_main_q_acoustic_cost;
  LaneMatrixView<InfoToken> d_main_q_info;
  LaneMatrixView<int2> d_aux_q_state_and_cost;
  LaneMatrixView<InfoToken> d_aux_q_info;
  LaneMatrixView<HashmapValueT> d_hashmap_values;
  LaneMatrixView<int2> h_list_final_tokens_in_main_q;
  LaneMatrixView<float2> d_main_q_extra_and_acoustic_cost;
  LaneMatrixView<int32> d_histograms;
  LaneMatrixView<int2> d_main_q_block_sums_prefix_sum;
  LaneMatrixView<int32> d_main_q_state_hash_idx;
  LaneMatrixView<int32> d_main_q_extra_prev_tokens_prefix_sum;
  LaneMatrixView<int32> d_main_q_n_extra_prev_tokens_local_idx;
  LaneMatrixView<InfoToken> d_main_q_extra_prev_tokens;

  int32 max_nlanes;   // nlanes_,最大的batch值
  int32 main_q_capacity, aux_q_capacity;
  CostType *d_arc_weights;  // fst中arc权重数组
  int32 *d_arc_nextstates;  // fst中下一个状态id数组
  int32 *d_arc_pdf_ilabels;  // fst中arc上ilabel的id数组
  uint32 *d_arc_e_offsets;   // fst中各状态emitting边的偏移
  uint32 *d_arc_ne_offsets;  // fst中各状态non-emitting边的偏移
  CostType *d_fst_final_costs;  // fst中Final costs；h_final_[i]表示状态i的final cost
  int32 nstates;                // fst中状态数
  CostType default_beam;        // 剪枝参数
  CostType lattice_beam;        // 剪枝参数
  int32 init_channel_id;        // 用于初始化的channel id
  StateId init_state;   //fst的初始状态
  CostType init_cost;  //初始cost
  int32 hashmap_capacity;  //等于main_q_capacity_
  int32 max_active;   //最大的激活的token数
  int32 adaptive_beam_static_segment;
  int32 adaptive_beam_bin_width;
};
// KernelParams 包含在内核调用之间更改的所有kernels参数
struct KernelParams {
  int32 nlanes_used;
};
```

### Counters

```cpp
// 保存单条语音解码的GPU状态
struct LaneCounters {
  // lane 将为当前帧计算的channel数
  ChannelId channel_to_compute;
  // 指向channel和当前帧的loglikelihoods数组指针
  BaseFloat *loglikelihoods;
  int2 main_q_narcs_and_end;
  int32 main_q_requested;
  int32 aux_q_requested;
  int32 aux_q_end;
  int32 post_expand_aux_q_end;  // 双buffer使用，
  //同一帧中的一些令牌共享相同的令牌数
  int32 main_q_n_extra_prev_tokens;
  // 在 emitting 状态创建的token数
  int32 main_q_n_emitting_tokens;
  // 队列是否溢出
  int32 q_overflow;
  int32 main_q_local_offset;
  int32 main_q_global_offset;
  int32 main_q_extra_prev_tokens_global_offset;
  // 帧对应的最小token
  IntegerCostType min_int_cost;
  IntegerCostType int_relative_cost;
  // beam可能和default_beam不同，因为AdaptiveBeam process或ApplyMaxActiveAndReduceBeam
  IntegerCostType int_beam;
  // Adaptive beam. 
  int2 adaptive_int_beam_with_validity_index;
  // min_cost + beam
  IntegerCostType int_cutoff;
  // The histogram for max_active will be computed between min_histo_cost
  // and max_histo_cost. Set for each frame after emitting stage
  CostType min_histo_cost;
  CostType max_histo_cost;
  CostType histo_bin_width;
  bool compute_max_active;
  // offsets used by concatenate_lanes_data_kernel
  int32 main_q_end_lane_offset;
  int32 main_q_n_emitting_tokens_lane_offset;
  int32 main_q_n_extra_prev_tokens_lane_offset;

  // --- Only valid after calling GetBestCost
  // min_cost and its arg. Can be different than min_cost, because we may
  // include final costs
  int2 min_int_cost_and_arg;  //{tokens中最小的cost，最小cost的token index}
  // Number of final tokens with cost < best + lattice_beam
  int32 n_within_lattice_beam;  //cost在 [min_cost; min_cost+lattice_beam]范围的token数
  int32 has_reached_final;  // final令牌队列是否包含至少一个与最终 FST 状态相关联的令牌
  int32 prev_arg_min_int_cost;
};

// 保存每条语音解码状态（CPU）
struct ChannelCounters {
  // All the following values are just saved values from LaneCounters
  // from the latest context-switch
  int2 prev_main_q_narcs_and_end;
  int32 prev_main_q_n_extra_prev_tokens;
  int32 prev_main_q_global_offset;
  int32 prev_main_q_extra_prev_tokens_global_offset;
  CostType prev_beam;
  // Only valid after calling GetBestCost
  // different than min_int_cost : we include the "final" cost
  int2 min_int_cost_and_arg_with_final;
  int2 min_int_cost_and_arg_without_final; //{min_int_cost,prev_arg_min_int_cost}
};
```



```cpp
class CudaDecoder {
 public:
  ///\param[in] fst A CudaFst instance. Not owned, must survive this object.
  ///\param[in] config
  ///\param[in] nlanes
  ///\param[in] nchannels
  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config, int32 nlanes,
              int32 nchannels);

  // Special constructor for nlanes = nchannels. Here for the non-advanced
  // user Here we can consider nchannels = batch size. If we want to
  // decode 10 utterances at a time, we can use nchannels = 10
  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config,
              int32 nchannels)
      : CudaDecoder(fst, config, nchannels, nchannels) {}
  virtual ~CudaDecoder() noexcept(false);

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);

  // 读取配置文件
  void ReadConfig(const CudaDecoderConfig &config);
  // InitDecoding 初始化解码，仅当您打算在 channels 中列出的通道上调用 AdvanceDecoding() 时才应使用
  void InitDecoding(const std::vector<ChannelId> &channels);
  // 计算 InitDecoding 的heavy H2H 副本。 通常在线程池上启动
  void InitDecodingH2HCopies(ChannelId ichannel);
  // 对一个批次进行解码（batch由N个channel组成）
  // batch中的数据进行同时解码，直到一个channel中没有可用于解码的数据
  // 解码完成后可以移除空的channel，也可以换入一个非空的channel进行解码
  // If max_num_frames is >= 0 it will decode no more than that many frames.
  void AdvanceDecoding(
      const std::vector<std::pair<ChannelId, BaseFloat *>> &lanes_assignements);

  // Version with deprecated API - will be removed at some point
  void AdvanceDecoding(const std::vector<ChannelId> &channels,
                       std::vector<CudaDecodableInterface *> &decodables,
                       int32 max_num_frames = -1);

  void AllowPartialHypotheses() {
    partial_traceback_ = generate_partial_hypotheses_ = true;
  }

  void AllowEndpointing() {
    if (frame_shift_seconds_ == FLT_MAX) {
      KALDI_ERR << "You must call SetOutputFrameShiftInSeconds() "
                << "to use endpointing";
    }
    partial_traceback_ = endpointing_ = true;
  }

  void SetOutputFrameShiftInSeconds(BaseFloat f) { frame_shift_seconds_ = f; }

  void GetPartialHypothesis(ChannelId ichannel, PartialHypothesis **out) {
    KALDI_ASSERT(generate_partial_hypotheses_);
    // No need to lock, all ops on h_all_channels_partial_hypotheses_out_ are
    // done before returning InitDecoding or AdvanceDecoding
    *out = &h_all_channels_partial_hypotheses_out_[ichannel];
  }

  bool EndpointDetected(ChannelId ichannel) {
    return h_all_channels_endpoint_detected_[ichannel];
  }

  // 返回给定channel中已解码的帧数
  int32 NumFramesDecoded(ChannelId ichannel) const;

  // 获取最佳解码路径. 
  // GetBestPath is deprecated and will be removed in a future release
  // For best path, use partial hypotheses
  void GetBestPath(const std::vector<ChannelId> &channels,
                   std::vector<Lattice *> &fst_out_vec,
                   bool use_final_probs = true);
  
  // ConcurrentGetRawLatticeSingleChannel()是线程安全版本的GetRawLattice，不在强依赖与CPU
  // 在控制线程调用PrepareForGetRawLattice，在执行线程调用ConcurrentGetRawLatticeSingleChannel
  // 如在CPU主线程调用PrepareForGetRawLattice（channel 8,6,3）
  // 在3个执行线程中分别执行ConcurrentGetRawLatticeSingleChannel的8,6,3的channel
  void PrepareForGetRawLattice(const std::vector<ChannelId> &channels,
                               bool use_final_probs);
  void ConcurrentGetRawLatticeSingleChannel(ChannelId ichannel,
                                            Lattice *fst_out);

  // 获取lattice解码路径 (using the lattice-beam in the CudaConfig parameters). 
  void GetRawLattice(const std::vector<ChannelId> &channels,
                     std::vector<Lattice *> &fst_out_vec, bool use_final_probs);

  // (可选) 解码是否使用CPU线程池
  // InitDecodingH2HCopies 和ComputeH2HCopies, 使用线程池
  void SetThreadPoolAndStartCPUWorkers(ThreadPoolLight *thread_pool,
                                       int32 nworkers);

  // Used to generate partial results
  void SetSymbolTable(const fst::SymbolTable &word_syms) {
    word_syms_ = &word_syms;
  }

 private:
  // 在构造函数中调用，用来申请内存
  void AllocateDeviceData();
  void AllocateHostData();
  void AllocateDeviceKernelParams();
  // 在构造函数中调用，用来初始化数据
  void InitDeviceData();
  void InitHostData();
  void InitDeviceParams();
  // 计算initial channel，为新语音创建一个initial channel
  void ComputeInitialChannel();
  // 将channels复制给channel_to_compute_
  // 设置h_kernel_params_->nlanes_used = channels.size();nlanes_used_ = channels.size();
  void SetChannelsInKernelParams(const std::vector<ChannelId> &channels);
  // 设置h_kernel_params_->nlanes_used = 0;  nlanes_used_ = 0;
  void ResetChannelsInKernelParams();
  // Context-switch functions
  // 加载/保存channel的状态到lane
  // 设置h_lanes_counters_状态
  void LoadChannelsStateToLanes(const std::vector<ChannelId> &channels);
  void SaveChannelsStateFromLanes();
  // 计算所有channel中准备好的最小帧数。如果max_num_frames>0，则再和max_num_frames比较，取最小
  int32 NumFramesToDecode(const std::vector<ChannelId> &channels,
                          std::vector<CudaDecodableInterface *> &decodables,
                          int32 max_num_frames);
  // 计算channels中各个channel的最佳cost
  // list_lattice_tokens保存所有cost在 [best; best+lattice_beam]范围的tokens
  // has_reached_final[ichannel] 保存所有到达最终状态的token
  void GetBestCost(
      const std::vector<ChannelId> &channels, bool isfinal,
      std::vector<std::pair<int32, CostType>> *argmins,
      std::vector<std::vector<std::pair<int, float>>> *list_lattice_tokens,
      std::vector<bool> *has_reached_final);

  // Fills *out_nonempty_channels with channels with NumFramesDecoded(ichannel)> 0
  void FillWithNonEmptyChannels(const std::vector<ChannelId> &channels,
                                std::vector<ChannelId> *out_nonempty_channels);

  // 给定一个token, 获取代价最小的前向节点
  // 在GetBestPath 或 best path traceback中使用
  void GetBestPredecessor(int32 ichannel, int32 curr_token_idx,
                          int32 *prev_token_idx_out, int32 *arc_idx_out);
  // 增加 emitting 状态的边
  void ExpandArcsEmitting();
  // 添加non-emitting 装填的边
  void ExpandArcsNonEmitting();
  // 当token数量大于max_active_，重新计算剪枝参数beam，是的token数量接近max_active_
  void ApplyMaxActiveAndReduceBeam(enum QUEUE_ID queue_id);
  // 在增加Emitting边时调用. 裁剪the aux_q (ExpandArcs的输出), 移动保留的tokens到main_q.
  void PruneAndPreprocess();
  // non-emitting完成, main_q是该帧的最终结果.
  //生成与该 main_q 关联的所有数据，例如列出共享相同令牌的不同令牌。next_state 我们还为下一帧的 ExpandArcsEmitting 进行预处理
  // PostProcessingMainQueue后，所有工作数据都恢复到原始状态，以确保我们为下一次上下文切换做好准备
  void PostProcessingMainQueue();
  // 拷贝main_q的数据到CPU内存,给GetBestPath/GetRawLattice使用
  // 在PostProcessingMainQueue更新数据后使用
  void CopyMainQueueDataToHost();
  // 检查队列是否溢出
  void CheckOverflow();
  // 计算所用lane中最长的最长边数
  int32 GetMaxForAllLanes(std::function<int32(const LaneCounters &)> func);
  // 同步/异步方式拷贝lane counters到CPU内存
  // lanes counters 包含as main_q_end(number of tokens in the main_q) main_q_narcs (number of arcs) 
  void CopyLaneCountersToHostAsync();
  void CopyLaneCountersToHostSync();
  // 每帧解码后的tokens会被拷贝到CPU内存，并保留在CPU内存,在解码到最后一帧时用于计算final lattice
  // 拷贝到CPU内存不是每个channel分开拷贝，而是将所用channel的结果拷贝到一个连续内存后再进行一次拷贝以提升性能
  // LaunchD2H and sLaunchH2HCopies将数据拷贝到CPU内存
  // ConcatenateData是将所有channel数据拷贝被一块连续内存
  void ConcatenateData();
  // Start the D2H copies used to send data back to host at the end of
  // each frames
  void LaunchD2HCopies();
  // ComputeH2HCopies
  // 将每帧解码后拷贝到CPU内存的数据进行unpack,赋值给CPU结构化数据
  void ComputeH2HCopies();

  // Used to generate the partial hypotheses
  // Called by the worker threads async
  void BuildPartialHypothesisOutput(
      ChannelId ichannel,
      std::stack<std::pair<int, PartialPathArc *>> *traceback_buffer_);
  void GeneratePartialPath(LaneId ilane, ChannelId ichannel);

  void EndpointDetected(LaneId ilane, ChannelId ichannel);
  // Wait for the async partial hypotheses related tasks to be done
  // before returning
  void WaitForPartialHypotheses();

  // Takes care of preparing the data for ComputeH2HCopies
  // and check whether we can use the threadpool or we have to do the work
  // on the current thread
  void LaunchH2HCopies();
  // Function called by the CPU worker threads
  // Calls ComputeH2HCopies when triggered
  void ComputeH2HCopiesCPUWorker();

  template <typename T>
  void MoveConcatenatedCopyToVector(const LaneId ilane,
                                    const ChannelId ichannel,
                                    const std::vector<int32> &lanes_offsets,
                                    T *h_concat,
                                    std::vector<std::vector<T>> *vecvec);
  void WaitForH2HCopies();
  void WaitForInitDecodingH2HCopies();
  // Computes a set of static asserts on the static values
  // In theory we should do them at compile time
  void CheckStaticAsserts();
  // Can be called in GetRawLattice to do a bunch of deep asserts on the
  // data Slow, so disabled by default
  void DebugValidateLattice();

  //
  // Data members
  //

  CudaDecoderConfig config_;
  const fst::SymbolTable *word_syms_;  // for partial hypotheses
  bool generate_partial_hypotheses_;   // set by AllowPartialHypotheses
  bool endpointing_;
  bool partial_traceback_;
  BaseFloat frame_shift_seconds_;

  std::set<int32> silence_phones_;

  // CudaFst 数据结构包含 CSR 格式的 FST 图，在 GPU 和 CPU 内存上
  const CudaFst& fst_;
  // 保存batch解码的每条语音的解码状态，保存在CPU内存中
  HostLaneMatrix<LaneCounters> h_lanes_counters_;
  // Counters of channels
  // Contains all the single values saved to remember the state of a
  // channel not used during computation. Those values are loaded/saved
  // into/from a lane during context switching
  ChannelCounters *h_channels_counters_;
  // Contain the various counters used by lanes/channels, such as
  // main_q_end, main_q_narcs. On device memory (equivalent of
  // h_channels_counters on device)
  DeviceChannelMatrix<ChannelCounters> d_channels_counters_;
  // lanes(batch)解码状态，保存在GPU内存中
  DeviceLaneMatrix<LaneCounters> d_lanes_counters_;
  // Number of lanes and channels, as defined in the constructor arguments
  int32 nlanes_, nchannels_;

  // We will now define the data used on the GPU
  // The data is mainly linked to two token queues
  // - the main queue
  // - the auxiliary queue
  //
  // The auxiliary queue is used to store the raw output of ExpandArcs.
  // We then prune that aux queue (and apply max-active) and move the
  // survival tokens in the main queue. Tokens stored in the main q can
  // then be used to generate new tokens (using ExpandArcs) We also
  // generate more information about what's in the main_q at the end of a
  // frame (in PostProcessingMainQueue)
  //
  // As a reminder, here's the data structure of a token :
  //
  // struct Token { state, cost, prev_token, arc_idx }
  //
  // Please keep in mind that this structure is also used in the context
  // of lattice decoding. We are not storing a list of forward links like
  // in the CPU decoder. A token stays an instanciation of an single arc.
  //
  // For performance reasons, we split the tokens in three parts :
  // { state } , { cost }, { prev_token, arc_idx }
  // Each part has its associated queue
  // For instance, d_main_q_state[i], d_main_q_cost[i], d_main_q_info[i]
  // all refer to the same token (at index i)
  // The data structure InfoToken contains { prev_token, arc_idx }
  // We also store the acoustic costs independently in
  // d_main_q_acoustic_cost_
  //
  // The data is eiher linked to a channel, or to a lane.
  //
  // Channel data (DeviceChannelMatrix):
  //
  // The data linked with a channel contains the data of frame i we need
  // to remember to compute frame i+1. It is the list of tokens from frame
  // i, with some additional info (ie the prefix sum of the emitting arcs
  // degrees from those tokens). We are only storing
  // d_main_q_state_and_cost_ as channel data because that's all we need
  // in a token to compute frame i+1. We don't need token.arc_idx or
  // token.prev_token. The reason why we also store that prefix sum is
  // because we do the emitting preprocessing at the end of frame i. The
  // reason for that is that we need infos from the hashmap to do that
  // preprocessing. The hashmap is always cleared at the end of a frame.
  // So we need to do the preprocessing at the end of frame i, and then
  // save d_main_q_degrees_prefix_sum_. d_main_q_arc_offsets is generated
  // also during preprocessing.
  //
  // Lane data (DeviceLaneMatrix):
  //
  // The lane data is everything we use during computation, but which we
  // reset at the end of each frame. For instance we use a hashmap at some
  // point during the computation, but at the end of each frame we reset
  // it. That way that hashmap is able to compute whichever channel the
  // next time AdvanceDecoding is called. The reasons why we do that is :
  //
  // - We use context switching. Before and after every frames, we can do
  // a context switching. Which means that a lane cannot save a channel's
  // state in any way once AdvanceDecoding returns. e.g., during a call of
  // AdvanceDecoding, ilane=2 may compute 5 frames from channel=57 (as
  // defined in the std::vector<ChannelId> channels). In the next call,
  // the same ilane=2 may compute 10 frames from channel=231. A lane data
  // has to be reset to its original state at the end of each
  // AdvanceDecoding call.
  // If somehow some data has to be saved, it needs to be declared as
  // channel data.
  //
  // - The reason why we make the distinction between lane and channel
  // data (in theory everything could be consider channel data), is
  // because a lane uses more memory than a channel. In the context of
  // online decoding, we need to create a lot channels, and we need them
  // to be as small as possible in memory. Everything that can be reused
  // between channels is stored as lane data.

  //
  // Channel data members:
  //
  // Matrix,每行表示一个channel解码后的tokens,行中每个元素为{state_id,cost}
  DeviceChannelMatrix<int2> d_main_q_state_and_cost_;
  // main_q 中arc的前缀总和. Used by ExpandArcs,
  // 在预处理阶段设置（PruneAndPreprocess 或 PostProcessingMainQueue 中的 preprocess_in_place）
  DeviceChannelMatrix<int32> d_main_q_degrees_prefix_sum_;
  // d_main_q_arc_offsets[i] = fst_.arc_offsets[d_main_q_state[i]]
  // token对应的边的arc_start
  DeviceChannelMatrix<int32> d_main_q_arc_offsets_;

  //
  // Lane data members:
  //

  // InfoToken
  // Usually contains {prev_token, arc_idx}
  // If more than one token is associated to a fst_state,
  // it will contain where to find the list of those tokens in
  // d_main_q_extra_prev_tokens
  // ie {offset,size} in that list. We differentiate the two situations by
  // calling InfoToken.IsUniqueTokenForStateAndFrame()
  DeviceLaneMatrix<InfoToken> d_main_q_info_;
  // token的声学分
  DeviceLaneMatrix<CostType> d_main_q_acoustic_cost_;
  // 在帧的末尾，我们使用哈希图来检测与相同 FST 状态 S 关联的令牌
  // 我们最后这样做，只在后修剪时使用哈希图
  // hashmap 中数据格式为{state_id,num_tokens,min_cost}
  DeviceLaneMatrix<HashmapValueT> d_hashmap_values_;
  // Reminder: in the GPU lattice decoder, a token is always associated
  // to a single arc. Which means that multiple tokens in the same frame
  // can be associated with the same FST state.
  //
  // We are NOT listing those duplicates as ForwardLinks in an unique
  // meta-token like in the CPU lattice decoder
  //
  // When more than one token is associated to a single FST state,
  // we will list those tokens into another list :
  // d_main_q_extra_prev_tokens we will also save data useful in such a
  // case, such as the extra_cost of a token compared to the best for that
  // state
  DeviceLaneMatrix<InfoToken> d_main_q_extra_prev_tokens_;   //一个fst state上有多个token时，用于存储tokens
  DeviceLaneMatrix<float2> d_main_q_extra_and_acoustic_cost_; //一个fst state上有多个token时，用于存储{extra_cost,acoustic_cost}
  // Histogram. Used to perform the histogram of the token costs
  // in the main_q. Used to perform a soft topk of the main_q (max-active)
  DeviceLaneMatrix<int32> d_histograms_;  
  // When filling the hashmap in PostProcessingMainQueue, we create a
  // hashmap value for each FST state presents in the main_q (if at least
  // one token is associated with that state)
  // d_main_q_state_hash_idx_[token_idx] is the index of the state
  // token.state in the hashmap Stored into a FSTStateHashIndex, which is
  // actually a int32. FSTStateHashIndex should only be accessed through
  // [Get|Set]FSTStateHashIndex, because it uses the bit sign to also
  // remember if that token is the representative of that state. If only
  // one token is associated with S, its representative will be itself
  DeviceLaneMatrix<FSTStateHashIndex> d_main_q_state_hash_idx_;
  // local_idx of the extra cost list for a state
  // For a given state S, first token associated with S will have
  // local_idx=0 the second one local_idx=1, etc. The order of the
  // local_idxs is random
  DeviceLaneMatrix<int32> d_main_q_n_extra_prev_tokens_local_idx_;
  // Where to write the extra_prev_tokens in the
  // d_main_q_extra_prev_tokens_ queue
  DeviceLaneMatrix<int32> d_main_q_extra_prev_tokens_prefix_sum_;
  // Used when computing the prefix_sums in preprocess_in_place. Stores
  // the local_sums per CTA
  DeviceLaneMatrix<int2> d_main_q_block_sums_prefix_sum_;
  // aux_q. Filled by ExpandArcs.
  // d_aux_q_state_and_cost_中tokens会被移动到main_q通过调用PruneAndPreprocess方法
  DeviceLaneMatrix<int2> d_aux_q_state_and_cost_;
  // aux tokeninfo队列，通过调用PruneAndPreprocess方法移动到main_q
  DeviceLaneMatrix<InfoToken> d_aux_q_info_;
  // Dedicated space for the concat of extra_cost. We should reuse memory
  DeviceLaneMatrix<float2> d_extra_and_acoustic_cost_concat_matrix_;
  DeviceLaneMatrix<InfoToken> d_extra_prev_tokens_concat_matrix_;
  DeviceLaneMatrix<CostType> d_acoustic_cost_concat_matrix_;
  DeviceLaneMatrix<InfoToken> d_infotoken_concat_matrix_;
  // We will list in d_list_final_tokens_in_main_q all tokens within
  // [min_cost; min_cost+lattice_beam] It is used when calling GetBestCost
  // We only use an interface here because we will actually reuse data
  // from d_aux_q_state_and_cost We are done using the aux_q when
  // GetBestCost is called, so we can reuse that memory
  HostLaneMatrix<int2> h_list_final_tokens_in_main_q_;
  // Parameters used by the kernels
  // DeviceParams contains all the parameters that won't change
  // i.e. memory address of the main_q for instance
  // KernelParams contains information that can change.
  // For instance which channel is executing on which lane
  std::unique_ptr<DeviceParams> h_device_params_;
  std::unique_ptr<KernelParams> h_kernel_params_;
  // 被组成lanes(batch)的channel id
  std::vector<ChannelId> channel_to_compute_;
  int32 nlanes_used_;  // number of lanes used in h_kernel_params_
  // Initial lane
  // When starting a new utterance,
  // init_channel_id is used to initialize a channel
  int32 init_channel_id_;
  // GPU解码使用的stream，compute_st_用来计算，copy_st_用来同步GPU和CPU内存数据
  cudaStream_t compute_st_, copy_st_;
  // Parameters extracted from CudaDecoderConfig
  // Those are defined in CudaDecoderConfig
  CostType default_beam_;
  CostType lattice_beam_;
  int32 ntokens_pre_allocated_;
  int32 max_active_;  // Target value from the parameters
  int32 aux_q_capacity_;
  int32 main_q_capacity_;
  // Hashmap capacity. Multiple of max_tokens_per_frame
  int32 hashmap_capacity_;
  // Static segment of the adaptive beam. Cf InitDeviceParams
  int32 adaptive_beam_static_segment_;
  // The first index of all the following vectors (or vector<vector>)
  // is the ChannelId. e.g., to get the number of frames decoded in
  // channel 2, look into num_frames_decoded_[2].

  // 已解码帧数
  std::vector<int32> num_frames_decoded_;
  // 每帧token在h_all_tokens_info_中的起始索引
  std::vector<std::vector<int32>> frame_offsets_;
  // Data storage. We store on host what we will need in
  // GetRawLattice/GetBestPath
  std::vector<std::vector<InfoToken>> h_all_tokens_info_;  //各channel中token info 集合
  std::vector<std::vector<CostType>> h_all_tokens_acoustic_cost_;  // token 对应的声学模型打分
  std::vector<std::vector<InfoToken>> h_all_tokens_extra_prev_tokens_; // 一个state上有多个token时，用于存储token
  std::vector<std::vector<float2>>
      h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_;   // h_all_tokens_extra_prev_tokens_中token对应的{extra_cost,acoustic_cost}
  //TODO(hugovbraun): At some point we should switch to a shared_lock to be
  // able to compute partial lattices while still streaming new data for this
  // channel.
  std::vector<std::mutex> channel_lock_;

  // For each channel, set by PrepareForGetRawLattice
  // argmin cost, list of the tokens within
  // [best_cost;best_cost+lattice_beam] and if we've reached a final
  // token. Set by PrepareForGetRawLattice.
  std::vector<std::pair<int32, CostType>> h_all_argmin_cost_;  //各channel中最小的cost{token_id,best_cost}
  std::vector<std::vector<std::pair<int, float>>> h_all_final_tokens_list_; //同list_finals_token_idx_and_cost_，{token_id,token_cost}
  std::vector<bool> h_all_has_reached_final_;  // 同 has_reached_final_
  // Buffer to store channels with nframes > 0.
  std::vector<ChannelId> nonempty_channels_;

  // Pinned memory arrays. Used for the DeviceToHost copies
  float2 *h_extra_and_acoustic_cost_concat_, *d_extra_and_acoustic_cost_concat_;  // 单帧解码后{extra_cost,acoustic_cost}数组
  InfoToken *h_infotoken_concat_, *d_infotoken_concat_;   //单帧解码后的token数组指针
  CostType *h_acoustic_cost_concat_, *d_acoustic_cost_concat_;   // 单帧解码后声学模型分数数组
  InfoToken *h_extra_prev_tokens_concat_, *d_extra_prev_tokens_concat_;  //单帧解码后，state上有多个token的情况下用于存储token的数组
  // second memory space used for double buffering
  float2 *h_extra_and_acoustic_cost_concat_tmp_;  //用于和GPU同步，会拷贝的到h_extra_and_acoustic_cost_concat_
  InfoToken *h_infotoken_concat_tmp_;
  CostType *h_acoustic_cost_concat_tmp_;
  InfoToken *h_extra_prev_tokens_concat_tmp_;
  // Offsets used in MoveConcatenatedCopyToVector
  std::vector<int32> h_main_q_end_lane_offsets_;   // 每帧产生的token在h_all_tokens_info_中的偏移
  std::vector<int32> h_emitting_main_q_end_lane_offsets_;  //每帧产生的emitting token在h_all_tokens_info_中的偏移
  std::vector<int32> h_n_extra_prev_tokens_lane_offsets_;
  // Index of the best index for the last frame. Used by endpointing/partial
  // results

  std::vector<BestPathTracebackHead> h_best_path_traceback_head_;
  std::vector<BestPathTracebackHead>
      h_all_channels_prev_best_path_traceback_head_;
  // Partial path so far on a given channel

  // Partial hypotheses to be used by user
  // Only valid between API calls (InitDecoding, AdvanceDecoding)
  std::vector<PartialHypothesis> h_all_channels_partial_hypotheses_out_;
  std::vector<char>
      h_all_channels_endpoint_detected_;  // not using a bool, we need it to be
                                          // threadsafe

  // Used internally to store the state of the current partial hypotheses
  std::vector<std::list<PartialPathArc>> h_all_channels_partial_hypotheses_;

  // Used when calling GetBestCost
  std::vector<std::pair<int32, CostType>> argmins_;  // 每个channel中cost的最小的token  {token_index,min_cost}
  std::vector<bool> has_reached_final_;  // 令牌队列是否包含至少一个与最终 FST 状态相关联的令牌
  std::vector<std::vector<std::pair<int32, CostType>>>
      list_finals_token_idx_and_cost_; //同h_all_final_tokens_list_,{token_id,token_cost},
                                       // token_cost在[best;best+lattice_beam]范围;token_id为在h_all_tokens_info_中的索引
  bool compute_max_active_;
  cudaEvent_t nnet3_done_evt_;
  cudaEvent_t d2h_copy_acoustic_evt_;
  cudaEvent_t d2h_copy_infotoken_evt_;
  cudaEvent_t d2h_copy_extra_prev_tokens_evt_;
  cudaEvent_t concatenated_data_ready_evt_;
  cudaEvent_t lane_offsets_ready_evt_;
  // GetRawLattice helper
  // Data used when building the lattice in GetRawLattice

  // few typedef to make GetRawLattice easier to understand
  // Returns a unique id for each (iframe, fst_state) pair
  // We need to be able to quickly identity a (iframe, fst_state) ID
  //
  // A lattice state is defined by the pair (iframe, fst_state)
  // A token is associated to a lattice state (iframe, token.next_state)
  // Multiple token in the same frame can be associated to the same
  // lattice state (they all go to the same token.next_state) We need to
  // quickly identify what is the lattice state of a token. We are able to
  // do that through GetLatticeStateInternalId(token), which returns the
  // internal unique ID for each lattice state for a token
  //
  // When we build the output lattice, we a get new lattice state
  // output_lattice_state = fst_out->AddState()
  // We call this one OutputLatticeState
  // The conversion between the two is done through maps
  // [curr|prev]_f_raw_lattice_state_
  typedef int32 LatticeStateInternalId;
  typedef StateId OutputLatticeState;
  typedef int32 TokenId;
  // 如果(frame,fst_state)只有一个token，则返回token_index,否则返回total_ntokens+token.prev_token
  LatticeStateInternalId GetLatticeStateInternalId(int32 total_ntokens,
                                                   TokenId token_idx,
                                                   InfoToken token);
  // Keeping track of a variety of info about states in the lattice
  // - token_extra_cost. A path going from the current lattice_state to
  // the end has an extra cost compared to the best path (which has an
  // extra cost of 0). token_extra_cost is the minimum of the extra_cost
  // of all paths going from the current lattice_state to the final frame.
  // - fst_lattice_state is the StateId of the lattice_state in fst_out
  // (in the output lattice). lattice_state is an internal state used in
  // GetRawLattice.
  // - is_state_closed is true if the token_extra_cost has been read by
  // another token. It means that the
  // token_extra_cost value has been used, and if we modify
  // token_extra_cost again, we may need to recompute the current frame
  // (so that everyone uses the latest token_extra_cost value)
  struct RawLatticeState {
    CostType token_extra_cost;
    OutputLatticeState fst_lattice_state;
    bool is_state_closed;
  };
  // extra_cost_min_delta_ used in the must_replay_frame situation. Please
  // read comments associated with must_replay_frame in GetRawLattice to
  // understand what it does
  CostType extra_cost_min_delta_;
  ThreadPoolLight *thread_pool_;
  std::vector<std::thread> cpu_dedicated_threads_;
  int32 n_threads_used_;
  std::vector<ChannelId> lanes2channels_todo_;  //lanes到channels的映射，index=lane，value=channel
  std::atomic<int> n_acoustic_h2h_copies_todo_;
  std::atomic<int> n_extra_prev_tokens_h2h_copies_todo_;
  //TODO(hugovbraun): unused: std::atomic<int> n_d2h_copies_ready_;
  std::atomic<int> n_infotoken_h2h_copies_todo_;
  int32 n_h2h_task_not_done_;
  int32 n_init_decoding_h2h_task_not_done_;
  std::atomic<int> n_h2h_main_task_todo_;
  std::mutex n_h2h_task_not_done_mutex_;
  std::mutex n_init_decoding_h2h_task_not_done_mutex_;
  std::mutex n_h2h_main_task_todo_mutex_;
  std::condition_variable n_h2h_main_task_todo_cv_;
  std::condition_variable h2h_done_;
  std::condition_variable init_decoding_h2h_done_;
  //TODO(hugovbraun): unused: std::atomic<bool> active_wait_;

  // Used for sync on partial hypotheses tasks
  std::atomic<std::int32_t> n_partial_traceback_threads_todo_;
  std::atomic<std::int32_t> n_partial_traceback_threads_not_done_;

  // Set to false in destructor to stop threads.
  volatile bool h2h_threads_running_;

  // Using the output from GetBestPath, we add the best tokens (as
  // selected in GetBestCost) from the final frame to the output lattice.
  // We also fill the data structures (such as q_curr_frame_todo_, or
  // curr_f_raw_lattice_state_) accordingly
  void AddFinalTokensToLattice(
      ChannelId ichannel,
      std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      Lattice *fst_out);
  // Check if a token should be added to the lattice. If it should, then
  // keep_arc will be true
  void ConsiderTokenForLattice(
      ChannelId ichannel, int32 iprev, int32 total_ntokens, TokenId token_idx,
      OutputLatticeState fst_lattice_start, InfoToken *tok_beg,
      float2 *arc_extra_cost_beg, CostType token_extra_cost,
      TokenId list_prev_token_idx, int32 list_arc_idx,
      InfoToken *list_prev_token, CostType *this_arc_prev_token_extra_cost,
      CostType *acoustic_cost, OutputLatticeState *lattice_src_state,
      bool *keep_arc, bool *dbg_found_zero);
  // Add the arc to the lattice. Also updates what needs to be updated in
  // the GetRawLattice datastructures.
  void AddArcToLattice(
      int32 list_arc_idx, TokenId list_prev_token_idx,
      InfoToken list_prev_token, int32 curr_frame_offset,
      CostType acoustic_cost, CostType this_arc_prev_token_extra_cost,
      LatticeStateInternalId src_state_internal_id,
      OutputLatticeState fst_lattice_start,
      OutputLatticeState to_fst_lattice_state,
      std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
      std::vector<std::pair<TokenId, InfoToken>> *q_prev_frame_todo,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *prev_f_raw_lattice_state,
      std::unordered_set<int32> *f_arc_idx_added, Lattice *fst_out,
      bool *must_replay_frame);
  // Read a token information
  void GetTokenRawLatticeData(
      TokenId token_idx, InfoToken token, int32 total_ntokens,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      CostType *token_extra_cost, OutputLatticeState *to_fst_lattice_state);

  // A token is an instance of an arc. It goes to a FST state
  // (token.next_state) Multiple token in the same frame can go to the
  // same FST state. GetSameFSTStateTokenList returns that list
  void GetSameFSTStateTokenList(ChannelId ichannel, InfoToken &token,
                                InfoToken **tok_beg,
                                float2 **arc_extra_cost_beg, int32 *nprevs);

  // Swap datastructures at the end of a frame. prev becomes curr (we go
  // backward)
  //
  void SwapPrevAndCurrLatticeMap(
      int32 iframe, bool dbg_found_best_path,
      std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
      std::vector<std::pair<TokenId, InfoToken>> *q_prev_frame_todo,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *prev_f_raw_lattice_state,
      std::unordered_set<int32> *f_arc_idx_added);
};
```

### InitDecoding

```cpp
void CudaDecoder::InitDecoding(const std::vector<ChannelId> &channels) {
  // 克隆init_channel_id_ channel 到 所有要初始化的channels
  const int nlanes_used = channels.size();
  // Getting *h_kernel_params ready to use
  LoadChannelsStateToLanes(channels);
  cudaMemcpyAsync(d_lanes_counters_.MutableData(), h_lanes_counters_.lane(0),
                  nlanes_used_ * sizeof(*h_lanes_counters_.lane(0)),
                  cudaMemcpyHostToDevice, compute_st_);

  // Size of the initial main_q
  ChannelCounters &init_channel_counters =
      h_channels_counters_[init_channel_id_];
  const int32 init_main_q_size =
      init_channel_counters.prev_main_q_narcs_and_end.y;

  KALDI_ASSERT(init_main_q_size > 0);
  // Getting the channels ready to compute new utterances
  InitDecodingOnDeviceKernel(
      KaldiCudaDecoderNumBlocks(init_main_q_size, nlanes_used),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  {
    std::lock_guard<std::mutex> n_h2h_not_done_lk(
        n_init_decoding_h2h_task_not_done_mutex_);
    n_init_decoding_h2h_task_not_done_ += channels.size();
  }
  for (ChannelId ichannel : channels) {
    ChannelCounters &channel_counters = h_channels_counters_[ichannel];
    channel_counters.prev_main_q_narcs_and_end =
        init_channel_counters.prev_main_q_narcs_and_end;
    channel_counters.prev_main_q_n_extra_prev_tokens =
        init_channel_counters.prev_main_q_n_extra_prev_tokens;
    channel_counters.prev_main_q_global_offset = 0;
    channel_counters.prev_main_q_extra_prev_tokens_global_offset = 0;
    channel_counters.prev_beam = default_beam_;

    int32 n_initial_tokens = h_all_tokens_info_[init_channel_id_].size();
    num_frames_decoded_[ichannel] = 0;
    h_channels_counters_[ichannel] = h_channels_counters_[init_channel_id_];
    h_all_argmin_cost_[ichannel] = {-1, 0.0f};
    frame_offsets_[ichannel].clear();
    frame_offsets_[ichannel].push_back(n_initial_tokens);
    if (thread_pool_)
      thread_pool_->enqueue(THREAD_POOL_HIGH_PRIORITY,
                            &CudaDecoder::InitDecodingH2HCopies, this,
                            ichannel);
    else
      InitDecodingH2HCopies(ichannel);
  }
}
```



### AdvanceDecoding

```cpp
void CudaDecoder::AdvanceDecoding(
    const std::vector<ChannelId> &channels,
    std::vector<CudaDecodableInterface *> &decodables, int32 max_num_frames) {
  if (channels.size() == 0) return;  // nothing to do
  // 上下文切换 : 加载 channels state 到lanes，设置h_lanes_counters_状态
  LoadChannelsStateToLanes(channels);
  KALDI_ASSERT(nlanes_used_ > 0);

  // 获取一个batch(lanes)中最短帧数作为待解码帧数
  int32 nframes_to_decode =
      NumFramesToDecode(channels, decodables, max_num_frames);

  // 对batch进行逐帧解码
  for (int32 iframe = 0; iframe < nframes_to_decode; ++iframe) {
    // 对batch中语音进行循环，从声学模型中获取每帧的状态打分
    for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
      ChannelId ichannel = channel_to_compute_[ilane];
      int32 frame = num_frames_decoded_[ichannel];
      h_lanes_counters_.lane(ilane)->loglikelihoods =
          decodables[ilane]->GetLogLikelihoodsCudaPointer(frame);
    }
    // 同步loglikelihoods到GPU内存
    cudaMemcpyAsync(d_lanes_counters_.MutableData(), h_lanes_counters_.lane(0),
                    nlanes_used_ * sizeof(*h_lanes_counters_.lane(0)),
                    cudaMemcpyHostToDevice, compute_st_);
    // compute_st_ will wait for nnet3 to complete
    cudaEventRecord(nnet3_done_evt_, cudaStreamPerThread);
    cudaStreamWaitEvent(compute_st_, nnet3_done_evt_, 0);

    // 使用上一帧的 argmin 估计截断值
    ResetForFrameAndEstimateCutoffKernel(
        KaldiCudaDecoderNumBlocks(1, nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
        compute_st_, *h_device_params_, *h_kernel_params_);
    // 重置max active状态。 如有必要，ApplyMaxActiveAndReduceBeam 会将其重新打开
    compute_max_active_ = false;

    // 处理发射边。 我们已经在上一帧的末尾完成了预处理阶段
    ExpandArcsEmitting();
    // 我们将循环直到令牌队列中有足够少的非发射弧。 然后退出循环
    for (int i = 0; i < KALDI_CUDA_DECODER_N_NON_EMITTING_MAIN_ITERATIONS;
         ++i) {
      // 如果 aux_q 之一包含超过 max_active_ 个令牌，我们将减少beam以仅保留 max_active_ 个令牌
      ApplyMaxActiveAndReduceBeam(AUX_Q);
      // 对aux_q进行剪枝. 应用最新的beam (如果被触发，则使用ApplyMaxActiveAndReduceBeam更新的beam)
      // 将survival tokens移动到main queue，并为下一个 ExpandArcs 进行必要的预处理
      PruneAndPreprocess();

      // "heavy duty" kernel for non-emitting. The long tail of small
      // non-emitting iterations will be done in
      // FinalizeProcessNonEmittingKernel
      ExpandArcsNonEmitting();
    }
    ApplyMaxActiveAndReduceBeam(AUX_Q);
    PruneAndPreprocess();
    // Finalizing process non emitting. Takes care of the long tail,
    // the final iterations with a small numbers of arcs. Do the work inside a
    // single CTA (per lane),
    FinalizeProcessNonEmittingKernel(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                                     KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
                                     compute_st_, *h_device_params_,
                                     *h_kernel_params_);

    // We now have our final token main queues for that frame

    // 对该帧的令牌进行后处理
    // - 进行下一个发射扩展所需的预处理（将在下一帧发生）
    // - 如果一个状态 S 有多个与之关联的标记，则生成这些标记的列表，它允许在 GetRawLattice 中有效地回溯
    // - 计算extra costs
    PostProcessingMainQueue();

    // Waiting on previous d2h before writing on same device memory
    cudaStreamWaitEvent(compute_st_, d2h_copy_extra_prev_tokens_evt_, 0);
    // 将要移动到主机的数据连接到大型阵列中
    ConcatenateData();
    // Copying the final lane counters for that frame
    CopyLaneCountersToHostSync();
    CheckOverflow();

    // 将GetRawLattice/GetBestPath需要的数据拷贝到host
    CopyMainQueueDataToHost();

    for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
      const ChannelId ichannel = channel_to_compute_[ilane];
      // We're done processing that frame
      ++num_frames_decoded_[ichannel];
      const int32 main_q_end =
          h_lanes_counters_.lane(ilane)->main_q_narcs_and_end.y;
      // Saving frame offsets for GetRawLattice
      frame_offsets_[ichannel].push_back(frame_offsets_[ichannel].back() +
                                         main_q_end);
    }
  }

  SaveChannelsStateFromLanes();
}
```

```cpp
void CudaDecoder::ExpandArcsEmitting() {
  ExpandArcsKernel<true>(KaldiCudaDecoderNumBlocks(nlanes_used_),
                         KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                         *h_device_params_, *h_kernel_params_);

  // Updating a few counters, like resetting aux_q_end to 0...
  // true is for IS_EMITTING
  PostExpandKernel<true>(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                         KALDI_CUDA_DECODER_ONE_THREAD_BLOCK, compute_st_,
                         *h_device_params_, *h_kernel_params_);
}

template <bool IS_EMITTING>
void ExpandArcsKernel(const dim3 &grid, const dim3 &block,
                      const cudaStream_t &st,
                      const DeviceParams &cst_dev_params,
                      const KernelParams &kernel_params) {
  expand_arcs_kernel<IS_EMITTING><<<grid, block, 0, st>>>(cst_dev_params,
                                                          kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}
template <bool IS_EMITTING>
void PostExpandKernel(const dim3 &grid, const dim3 &block,
                      const cudaStream_t &st,
                      const DeviceParams &cst_dev_params,
                      const KernelParams &kernel_params) {
  post_expand_kernel<IS_EMITTING><<<grid, block, 0, st>>>(cst_dev_params,
                                                          kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}
```









## BatchedThreadedNnet3CudaPipelineConfig

```cpp
struct BatchedThreadedNnet3CudaPipelineConfig {
  BatchedThreadedNnet3CudaPipelineConfig()
      : max_batch_size(200),
        num_channels(-1),
        batch_drain_size(10),
        num_control_threads(2),
        num_worker_threads(20),
        determinize_lattice(true),
        max_pending_tasks(4000),
        pending_queue_padding(10),
        num_decoder_copy_threads(2),
        gpu_feature_extract(true) {};
  int max_batch_size;
  int num_channels;
  int batch_drain_size;
  int num_control_threads;
  int num_worker_threads;
  bool determinize_lattice;
  int max_pending_tasks;
  int pending_queue_padding;
  int num_decoder_copy_threads;
  bool gpu_feature_extract;

  void ComputeConfig() {
    if (num_channels == -1)
      num_channels =
          max_batch_size * KALDI_CUDA_DECODER_CHANNELS_BATCH_SIZE_RATIO;
  }

  OnlineNnet2FeaturePipelineConfig feature_opts;      // constant readonly
  CudaDecoderConfig decoder_opts;                     // constant readonly
  fst::DeterminizeLatticePhonePrunedOptions det_opts; // constant readonly
  nnet3::NnetBatchComputerOptions compute_opts;       // constant readonly
};
```

## BatchedThreadedNnet3CudaPipeline

```cpp
class BatchedThreadedNnet3CudaPipeline {
 public:
  BatchedThreadedNnet3CudaPipeline(
      const BatchedThreadedNnet3CudaPipelineConfig &config)
      : config_(config), all_group_tasks_not_done_(0) {
    config_.ComputeConfig();
  };

  // 加载fst 
  // 创建控制线程（ExecuteWorker）
  void Initialize(const fst::Fst<fst::StdArc> &decode_fst,
                  const nnet3::AmNnetSimple &nnet,
                  const TransitionModel &trans_model);

  // deallocates reusable objects
  void Finalize();

  // query a specific key to see if compute on it is complete
  bool isFinished(const std::string &key);

  // remove an audio file from the decoding and clean up resources
  void CloseDecodeHandle(const std::string &key);
  void CloseAllDecodeHandlesForGroup(const std::string &group);
  void CloseAllDecodeHandles();

  // 创建tack，并将task提交到decoding task队列pending_task_queue_
  void OpenDecodeHandle(
      const std::string &key, const WaveData &wave_data,
      const std::string &group = std::string(),
      const std::function<void(CompactLattice &clat)> &callback =
          std::function<void(CompactLattice &clat)>());
  // When passing in a vector of data, the caller must ensure the data
  // exists until the CloseDecodeHandle is called
  void OpenDecodeHandle(
      const std::string &key, const VectorBase<BaseFloat> &wave_data,
      float sample_rate, const std::string &group = std::string(),
      const std::function<void(CompactLattice &clat)> &callback =
          std::function<void(CompactLattice &clat)>());

  // Copies the raw lattice for decoded handle "key" into lat
  bool GetRawLattice(const std::string &key, Lattice *lat);
  // Determinizes raw lattice and returns a compact lattice
  bool GetLattice(const std::string &key, CompactLattice *lat);

  int32 GetNumberOfTasksPending();

  // Wait for all tasks to complete
  void WaitForAllTasks();
  // Wait for all tasks in the group to complete
  void WaitForGroup(const std::string &group);
  // Check if a group is available. Returns if not.
  bool IsGroupCompleted(const std::string &group);
  // Wait for any group to complete, then returns which group completed
  std::string WaitForAnyGroup();
  // Check if any group is available. If one is available, set its name in
  // *group
  bool IsAnyGroupCompleted(std::string *group);
  inline int NumPendingTasks() {
    return (tasks_back_ - tasks_front_ + config_.max_pending_tasks +
            config_.pending_queue_padding) %
           (config_.max_pending_tasks + config_.pending_queue_padding);
  };

 private:
  // Task data used during computation
  // Is cleared when task is completed
  struct TaskData {
    Vector<BaseFloat> raw_data;  // Wave 数据
    std::shared_ptr<SubVector<BaseFloat>> wave_samples;  // wave数据中的一段数据的指针
    float sample_frequency;  //采样率
    Vector<BaseFloat> ivector_features_cpu;  // 存放ivector的CPU特征
    Matrix<BaseFloat> input_features_cpu;  // 存放mfcc/fbank的CPU特征
    CuVector<BaseFloat> ivector_features;  // 存放ivector的GPU特征
    CuMatrix<BaseFloat> input_features;    // 存放mfcc/fbank的GPU特征
    CuMatrix<BaseFloat> posteriors;  
    
    //将多通道的语音整理到Vector中，并将整理后的数据头指针赋值给wave_samples，并给sample_frequency赋值
    TaskData(const WaveData &wave_data_in): wave_samples(NULL), sample_frequency(0) {
      int rows = wave_data_in.Data().NumRows();
      int cols = wave_data_in.Data().NumCols();
      int stride = wave_data_in.Data().Stride();

      raw_data.Resize(rows * cols, kUndefined);

      if (stride == cols) {
        // contigious so use one large memory copy
        memcpy(raw_data.Data(), wave_data_in.Data().Data(),
               rows * cols * sizeof(BaseFloat));
      } else {
        // data is not contigious so we need to copy one
        // row at a time
        for (int i = 0; i < rows; i++) {
          memcpy(raw_data.Data() + i * cols, wave_data_in.Data().RowData(i),
                 cols * sizeof(BaseFloat));
        }
      }
      wave_samples =
          std::make_shared<SubVector<BaseFloat>>(raw_data, 0, raw_data.Dim());
      sample_frequency = wave_data_in.SampFreq();
    };

    // 使用语音数据创建TaskData。 浅拷贝
    TaskData(const VectorBase<BaseFloat> &wave_data_in, float sample_rate) {
      wave_samples = std::make_shared<SubVector<BaseFloat>>(wave_data_in, 0,
                                                            wave_data_in.Dim());
      sample_frequency = sample_rate;
    }
  };

  // State needed for each decode task.
  // This state can be passed around by reference or pointer safely
  // and provides a convieniet way to store all decoding state.
  struct TaskState {
    std::string key;   // Task的唯一标识
    std::string group;  // group for that task. "" is default
    bool error;  
    std::string error_string;

    std::unique_ptr<TaskData> task_data;  //语音数据

    int32 ichannel;              // associated CudaDecoder channel
    Lattice lat;                 // Raw Lattice output
    CompactLattice dlat;         // Determinized lattice output.  Only set
                                 // if determinize-lattice=true
    std::atomic<bool> finished;  // Tells master thread if task has
                                 // finished execution

    bool determinized;

    // 任务完成后的回调函数（可选），在任务完成后，可通过解码出的lattice计算最佳路径等
    std::function<void(CompactLattice &clat)> callback;

    TaskState() : error(false), finished(false), determinized(false) {}

    // 使用语音数据初始化TaskState，数据会被拷贝为深拷贝
    void Init(const std::string &key_in, const WaveData &wave_data_in) {
      task_data.reset(new TaskData(wave_data_in));
      key = key_in;
    };
    // 使用语音数据初始化TaskState，数据会被拷贝为浅拷贝
    void Init(const std::string &key_in,
              const VectorBase<BaseFloat> &wave_data_in, float sample_rate) {
      task_data.reset(new TaskData(wave_data_in, sample_rate));
      key = key_in;
    }
  };

  // 通过key和group创建一个任务，并将任务和任务状态记录到tasks_lookup_的map中提供查询
  TaskState *AddTask(const std::string &key, const std::string &group);

  // 一个batch中各个语音解码状态
  struct ChannelState {  //typedef int32 ChannelId;
    std::vector<ChannelId> channels;   // 一个batch中待解码的语音特征集合
    std::vector<ChannelId> free_channels;  
    std::vector<ChannelId> completed_channels; //一个batch中解码完成的语音特征集合
    std::mutex free_channels_mutex;
  };

  // 将task添加到pending_task_queue_
  void AddTaskToPendingTaskQueue(TaskState *task);

  // 从任务队列中获取一个batch的Task。可能不足一个batch 
  void AquireAdditionalTasks(CudaDecoder &cuda_decoder,
                             ChannelState &channel_state,
                             std::vector<TaskState *> &tasks);

  // Computes Features for a single decode instance.
  void ComputeOneFeatureCPU(TaskState *task);

  // Computes MFCC/Fbank features across the tasks[first,tasks.size()
  void ComputeBatchFeatures(int32 first, std::vector<TaskState *> &tasks,
                            OnlineCudaFeaturePipeline &feature_pipeline);

  // Computes Nnet across the current decode batch
  void ComputeBatchNnet(nnet3::NnetBatchComputer &computer, int32 first,
                        std::vector<TaskState *> &tasks);

  // Allocates decodables for tasks in the range of
  // dstates[first,dstates.size())
  void AllocateDecodables(int32 first, std::vector<TaskState *> &tasks,
                          std::vector<CudaDecodableInterface *> &decodables);

  // Removes all completed channels from the channel list.
  // Also enqueues up work for post processing
  void RemoveCompletedChannels(
      CudaDecoder &cuda_decoder, ChannelState &channel_state,
      std::vector<CudaDecodableInterface *> &decodables,
      std::vector<TaskState *> &tasks);

  // For each completed decode perform post processing work and clean up
  void PostDecodeProcessing(CudaDecoder &cuda_decoder,
                            ChannelState &channel_state,
                            std::vector<CudaDecodableInterface *> &decodables,
                            std::vector<TaskState *> &tasks);

  // Calls ConcurrentGetRawLatticeSingleChannel and Determinize
  // on a dedicated CPU worker thread at the end of the decode
  void CompleteTask(CudaDecoder *cuda_decoder, ChannelState *channel_state,
                    TaskState *state);

  // Determinize one lattice
  void DeterminizeOneLattice(TaskState *task);
  // 控制线程执行函数；
  //1. 从任务列表中获取一个batch的数据
  //2. 对一个批次进行解码
  //3. 获取解码结果
  void ExecuteWorker(int threadId);

  BatchedThreadedNnet3CudaPipelineConfig config_;

  std::unique_ptr<CudaFst> cuda_fst_;
  const TransitionModel *trans_model_;
  const nnet3::AmNnetSimple *am_nnet_;
  OnlineNnet2FeaturePipelineInfo *feature_info_;

  std::mutex tasks_mutex_;         // protects tasks_front_ and
                                   // pending_task_queue_ for workers
  std::mutex tasks_add_mutex_;     // protect OpenDecodeHandle if multiple
                                   // threads access
  std::mutex tasks_lookup_mutex_;  // protext tasks_lookup map
  std::condition_variable tasks_lookup_cv_;
  std::atomic<int> tasks_front_, tasks_back_;  //任务队列（pending_task_queue_）的头、尾索引
  TaskState **pending_task_queue_;  // TaskState指针数据的头指针

  std::atomic<bool> exit_;       // signals threads to exit
  std::atomic<int> numStarted_;  // signals master how many threads have started

  ThreadPool *work_pool_;  // thread pool for CPU work  工作线程池
  std::map<std::string, int32> group_tasks_not_done_;  //任务分组中未完成任务数
  int32 all_group_tasks_not_done_;  //所有未完成任务数
  std::mutex group_tasks_mutex_;  
  std::condition_variable group_done_cv_;
  std::unordered_multimap<std::string, TaskState *> tasks_group_lookup_;  // 各个任务分组中的任务状态
  std::unordered_map<std::string, TaskState> tasks_lookup_; //保存Task的key和TaskState
  std::vector<std::thread> thread_contexts_;  // A list of thread contexts
};
```







```cpp
// host矩阵，使用cudaMallocHost分配pinned memory(不会交换到磁盘)，通过zero-copy功能映射到设备地址空间，从GPU直接访问，省掉主存与显存间进行数据拷贝的工作
template <typename T>
class HostMatrix {
  T *data_;   //数据起始指针
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ncols_ > 0);
    KALDI_ASSERT(!data_);
    CU_SAFE_CALL(cudaMallocHost((void **)&data_, (size_t)nrows_ * ncols_ * sizeof(*data_)));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    CU_SAFE_CALL(cudaFreeHost(data_));
  }

 protected:
  int32 ncols_;
  int32 nrows_;

 public:
  HostMatrix() : data_(NULL), ncols_(0), nrows_(0) {}

  virtual ~HostMatrix() {
    if (data_) Free();
  }
  // 修改矩阵大小，重新分配内存
  void Resize(int32 nrows, int32 ncols) {
    if (data_) Free();
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ncols > 0);
    nrows_ = nrows;
    ncols_ = ncols;
    Allocate();
  }
  T *MutableData() {
    KALDI_ASSERT(data_);
    return data_;
  }
};
//device上的矩阵，直接分配显存
template <typename T>
class DeviceMatrix {
  T *data_;
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ncols_ > 0);
    KALDI_ASSERT(!data_);
    data_ = static_cast<T *>(CuDevice::Instantiate().Malloc(
        (size_t)nrows_ * ncols_ * sizeof(*data_)));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    CuDevice::Instantiate().Free(data_);
  }

 protected:
  int32 ncols_;
  int32 nrows_;

 public:
  DeviceMatrix() : data_(NULL), ncols_(0), nrows_(0) {}

  virtual ~DeviceMatrix() {
    if (data_) Free();
  }

  void Resize(int32 nrows, int32 ncols) {
    if (data_) Free();
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ncols > 0);
    nrows_ = nrows;
    ncols_ = ncols;
    Allocate();
  }
  T *MutableData() {
    KALDI_ASSERT(data_);
    return data_;
  }
};
```



```cpp

```

```cpp

```









