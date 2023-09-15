# kaldi源码解析10-cudamatrix

cudamatrix包括设备管理，显存管理，数组/矩阵运算

## cu-device

cu-device包含用于选择CUDA设备，初始化cuBLAS和cuSparse句柄以及提供用于内存分配的接口（支持缓存，以避免CUDA内存分配器慢速）。

程序的每个线程都有一个单独的CuDevice对象instance，但是其许多变量是静态的（因此，在所有实例之间共享）。

（当前）仅支持使用单个GPU设备。 但是支持多个CUDA流。 这里的预期编程模型是有多个CPU线程，并且每个CPU线程都会自动获取自己的CUDA流，使用-DCUDA_API_PER_THREAD_DEFAULT_STREAM进行编译。

在同步多个线程的活动方面：CuDevice对象（在基础CuAllocator对象的帮助下）确保内存缓存代码本身不会成为同步问题的原因。  也就是说，不必担心当您使用CuDevice :: Malloc（）进行分配时，该内存仍将被GPU上的另一个线程使用。 但是，有时仍可能需要通过调用函数SynchronizeGpu（）来同步多个流的活动（可能恰好在线程增加信号量之前，紧接在等待信号量之后，或在获取互斥量之后，或类似的。）

```cpp
class CuDevice {
 public:

  // 单例模式，如果想要使用GPU，在程序的开头使用
  // CuDevice::Instantiate().SelectGpuId(..).
  static inline CuDevice& Instantiate() {
    CuDevice &ans = this_thread_device_;
    if (!ans.initialized_)
      ans.Initialize();
    return ans;
  }

  inline cublasHandle_t GetCublasHandle() { return cublas_handle_; }
  inline cusparseHandle_t GetCusparseHandle() { return cusparse_handle_; }
  inline curandGenerator_t GetCurandHandle() { return curand_handle_; }
  inline cusolverDnHandle_t GetCusolverDnHandle() { 
#if CUDA_VERSION < 9010
    KALDI_ERR << "CUDA VERSION '" << CUDA_VERSION << "' not new enough to support "
      << "cusolver. Upgrade to at least 9.1";
#endif
    return cusolverdn_handle_; 
  }

  inline void SeedGpu() {
    if (CuDevice::Instantiate().Enabled()) {
      // 设置随机种子，使srand()获得相同的序列
      CURAND_SAFE_CALL(curandSetPseudoRandomGeneratorSeed(
            curand_handle_, RandInt(128, RAND_MAX)));
      CURAND_SAFE_CALL(curandSetGeneratorOffset(curand_handle_, 0));
    }
  }
  //kaldi提供函数Malloc（），MallocPitch（）和Free（）来替换cudaMalloc（），cudaMallocPitch（）和cudaFree（）。 
  //它们的功能是缓存先前分配的结果，以避免CUDA分配内存带来的时间开销。
  inline void* Malloc(size_t size) {
    return multi_threaded_ ? g_cuda_allocator.MallocLocking(size) :
        g_cuda_allocator.Malloc(size);
  }

  inline void* MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch) {
    if (multi_threaded_) {
      return g_cuda_allocator.MallocPitchLocking(row_bytes, num_rows, pitch);
    } else if (debug_stride_mode_) {
      // The pitch bucket size is hardware dependent.
      // It is 512 on K40c with CUDA 7.5
      // "% 8" ensures that any 8 adjacent allocations have different pitches
      // if their original pitches are same in the normal mode.
      return g_cuda_allocator.MallocPitch(
          row_bytes + 512 * RandInt(0, 4), num_rows,
          pitch);
    } else {
      return g_cuda_allocator.MallocPitch(row_bytes, num_rows, pitch);
    }
  }

  inline void Free(void *ptr) {
    if (multi_threaded_) g_cuda_allocator.FreeLocking(ptr);
    else g_cuda_allocator.Free(ptr);
  }

  /// 选择一个GPU进行计算。 只在程序的开头（从主线程）调用一次此函数，或者根本不调用此函数。
  /// The 'use_gpu' modes are:
  ///  "yes" -- 自动选择GPU，如果失败则死亡。 如果将GPU设置为独占模式，它将随机选择一个；否则，它将选择具有最大可用内存的那个
  //  建议将GPU设置为独占模式，或者通过将变量CUDA_VISIBLE_DEVICES设置为您要使用程序的GPU的ID来控制要使用的GPU。
  ///  "optional" -- 和yes操作相同, 但失败会选择 CPU.
  ///  "no"       -- 在CPU运行.
  void SelectGpuId(std::string use_gpu);

  // 选择一个特定的GPU进行计算。 将为该设备重用现有的Cuda上下文。 初始化GPU使用所需的句柄（例如cublas句柄）
  bool SelectAndInitializeGpuIdWithExistingCudaContext(int dev_id);

  /// 检查是否选择使用CUDA GPU
  bool Enabled() const {
    return (device_id_ > -1);
  }

  /// 如果我们没有GPU或我们有GPU并且它支持双精度，则返回true。
  bool DoublePrecisionSupported();

  ///此函数会累积有关计时的统计信息，这些统计信息将在您调用PrintProfile（）时打印出来。 
  //但是，仅当VerboseLevel（）> = 1时，它才会执行某些操作。
  void AccuProfile(const char *function_name, const CuTimer &timer);

  /// 使用KALDI_LOG打印一些性能分析信息.
  void PrintProfile();

  /// 使用KALDI_LOG打印一些内存使用信息
  void PrintMemoryUsage() const;

  /// 如果程序计划从多个线程访问GPU（例如通过使用CuMatrix类），则用户应调用此函数。 
  //如果您无法为多线程程序调用此函数，则它可能会偶尔出现段错误（并且代码还会检测到您未能调用它，并会显示警告）。
  inline void AllowMultithreading() { multi_threaded_ = true; }

  /// 获取GPU名称
  void DeviceGetName(char* name, int32 len, int32 dev);

  /// 通过将GPU + CPU上的小矩阵相乘来检查GPU是否处于良好状态。 GPU过热可能导致结果不准确，我们希望检测到.
  void CheckGpuHealth();

  /// 如果Enabled（），则返回字节数n，以使矩阵步长始终是n的倍数（来自properties_.textureAlignment）。 
  //否则，返回16，这是用于CPU矩阵的步幅。
  int32 GetMatrixAlignment() const;

  /// 调用SetDebugStrideMode（true）激活一种模式，在这种模式下，对MallocPitch的调用将有目的地分配具有不同音调的数组
  //（两次调用之间不一致）。 这仅对测试代码有用。 此函数返回先前的模式，其中true表示音高不一致。
  bool SetDebugStrideMode(bool mode) {
    bool old_mode = debug_stride_mode_;
    debug_stride_mode_ = mode;
    return old_mode;
  }

  /// 检查GPU是否是性能模式
  bool IsComputeExclusive();

  // 注册CUDA设备的命令行选项.  
  // 必须在 CuDevice::Initialize()之前
  static void RegisterDeviceOptions(OptionsItf *po) {
    CuDevice::device_options_.Register(po);  
  }
  ~CuDevice();
 private:

  struct CuDeviceOptions {
    bool use_tensor_cores; // Enable tensor cores
    CuDeviceOptions () : use_tensor_cores(false) {};
    void Register(OptionsItf *po) {
      po->Register("cuda-use-tensor-cores", &use_tensor_cores, 
          "Enable FP16 tensor math. "
          "This is higher performance but less accuracy. "
          "This is only recommended for inference.");
    }
  };

  static CuDeviceOptions device_options_;

  // Default constructor used to initialize this_thread_device_
  CuDevice();
  CuDevice(CuDevice&); // Disallow.
  CuDevice &operator=(CuDevice&);  // Disallow.


  /// 在使用GPU的的情况下，在除主线程之外的其他线程中执行以下操作：
  //调用cudaSetDevice（）并设置cublas_handle_和cusparse_handle_
  void Initialize();

  /// 自动选择GPU并获取CUDA上下文（如果GPU处于非独占模式，则只能从SelectGpuId（）调用它）。 成功返回真.
  bool SelectGpuIdAuto();

  // 根据其ID选择GPU。 从SelectGpuIdAuto或SelectGpuIdWithExistingCudaContext调用
  bool SelectGpuId(int dev_id);

  /// 当存在与我们要使用的GPU对应的GPU上下文时，从SelectGpuId（）调用的此函数； 
  //它计算出设备ID，创建cuBLAS和cuSparse句柄，并打印出一些对调试有用的信息。 
  //它还将initialized_设置为true，以防止将来在主线程上调用Initialize（），因为这将尝试再次创建句柄。
  void FinalizeActiveGpu();

  /// Should only be called if Enabled() == true.
  int32 MajorDeviceVersion();

  /// Should only be called if Enabled() == true.
  int32 MinorDeviceVersion();


  // 每个线程都有自己的CuDevice对象，其中包含cublas和cusparse句柄。 这些对于线程是唯一的（这是NVidia推荐的）。
  static thread_local CuDevice this_thread_device_;

  // 使用的GPU设备ID。 这将初始化为-1，并在用户从主线程调用CuDevice :: Instantiate :: SelectGpuId（...）时进行设置
  static int32 device_id_;

  // 如果应用程序具有多个访问GPU设备的线程，则它将自动设置为true。 它用于标志在访问分配器和与分析相关的代码时是否使用锁。
  static bool multi_threaded_;

  // 仅在verbose level> = 1时才使用变量profile_map_。 它将积累一些功能级别的时序信息，这些信息将在程序结束时打印出来。 
  //这使程序变慢了，因为我们必须调用cudaDeviceSynchronize（）使时序信息有意义。
  static unordered_map<std::string, double, StringHasher> profile_map_;
  // profile_mutex_在多线程_为true的情况下监护profile_map _.
  static std::mutex profile_mutex_;

  // free_memory_at_startup_仅用于打印设备使用的内存。
  static int64 free_memory_at_startup_;
  static cudaDeviceProp properties_;

  // 如果etDebugStrideMode()设置为true，则将在分配数据时激活代码以使用伪随机跨度值（以检测错误，否则这种错误将很少发生）
  static bool debug_stride_mode_;


  // 初始化为false；如果用户在仍为false的线程中调用Instantiate()，则将调用Initialize()，
  //以便调用cudaSetDevice()并设置cublas和cusparse句柄
  bool initialized_;

  // 此变量只是静态变量device_id_的副本。 它用于检测何时以错误的方式调用此代码。
  int32 device_id_copy_;

  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;
  curandGenerator_t curand_handle_;
  cusolverDnHandle_t cusolverdn_handle_;
}; // class CuDevice
```

```cpp
//在需要同步线程的地方调用
void SynchronizeGpu();
```

## cu-allocator

CuMemoryAllocator类从GPU分配较大的内存区域，并为用户分配其子块。 这是必需的，因为CUDA malloc和free例程非常慢。

用户不直接访问此类，而是通过CuDevice对象访问它。 CuDevice类使用此类的Malloc（）和MallocPitch（）函数分配内存，并使用其Free（）函数释放它们，并且此类缓存内存块，以避免过于频繁地调用CUDA库的malloc / free函数。 如果应用程序使用多个线程，则必须在使用该类之前将其锁定，在这种情况下，CuDevice类将调用分配函数的MallocLocking（）和MallocPitchLocking（）版本（但用户应调用CuDevice :: AllowMultithreading（ ）（如果应用程序计划使用来自多个CPU线程的GPU功能）。

同步注意事项：如果使用多个CUDA流，则任何缓存分配器在CUDA流之间共享其池时可能会出现问题。也就是说：如果内存块被流1释放并分配给了流2，则操作可能在流1完成对该存储位置的工作之前在流2中开始。我们在这里使用技术含量较低的解决方案来解决此问题，它依赖于调用SynchronizeGpu（）来将no-op内核提交到旧版默认流中。每次调用CuMemoryAllocator（）:: Free（）并在此类中缓存内存块时，我们都会记录从中释放了该内存块的CPU线程的线程ID以及时间戳（CuMemoryAllocator的t_成员，每次使用该类时我们都会增加）。当我们分配缓存的内存时，我们尝试从同一CPU线程重新分配的块中分配内存；如果那是不可能的，并且自从块释放以来我们没有调用SynchronizeGpu（），那么我们调用SynchronizeGpu（）。希望这种情况很少发生。请注意，这是基于用户正在使用每线程默认流的假设（实际上，这就是我们的编译方式）。如果用户要明确使用CUDA流，则此机制不一定足以防止数据争用情况，并且用户可能必须采取进一步的预防措施。

关于碎片化的注意事项：内存碎片化是使用此类分配器时会遇到的主要问题之一。该分配器将分配少量的大型内存区域，并分配需要的从区域中拆分出的较小内存。当用户释放内存时，它将始终尽可能多地合并相邻的块。避免过多内存碎片的主要试探法是，它总是尽可能在尽可能接近内存区域起始位置的内存中进行分配。这将倾向于将所有较小的分配放在内存区域的开头，并希望在结尾处保留大块。始终尽可能从内存区域的起始位置开始分配的机制是，我们将内存区域划分为少量的子区域，并且在处理分配请求时，从编号最小的区域开始分配可以满足该大小要求的子区域。 （注意：我们可以分配跨越子区域的块，因此这种方法并不限制我们可以分配的块大小）。

```cpp
class CuMemoryAllocator {
 public:
  /// 申请GPU内存
  void* Malloc(size_t size);

  /// Allocation function for matrix-like things.
  void* MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch);

  /// Free device memory allocated by Malloc() or MallocPitch().
  void Free(void *ptr);

  /// Malloc（）的互斥体保护版本，用于多线程程序。
  inline void* MallocLocking(size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
    return Malloc(size);
  }
  /// MallocPitch（）的互斥体保护版本，用于多线程程序。
  inline void* MallocPitchLocking(size_t row_bytes, size_t num_rows, size_t *pitch) {
    std::unique_lock<std::mutex> lock(mutex_);
    return MallocPitch(row_bytes, num_rows, pitch);
  }
  ///Free()的互斥体保护版本，用于多线程程序
  void FreeLocking(void *ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    Free(ptr);
  }

  void PrintMemoryUsage() const;

  // 返回缓存中分配的当前内存
  size_t GetAllocatedMemory() { return allocated_memory_; }

  //  返回当前执行期间缓存中使用的最大内存
  size_t GetMaxAllocatedMemory() { return max_allocated_memory_; }

  CuMemoryAllocator();

  // Allows you to set options: must be called before any Malloc function is
  // called on this class.  It's done this way so the options can be changed
  // by the user (c.f. RegisterCuAllocatorOptions()) before the options are read.
  void SetOptions(const CuAllocatorOptions &opts) { opts_ = opts; }

  ~CuMemoryAllocator();

 private:

  struct SubRegion;

  struct MemoryBlock {
    char *begin;  // The beginning of the block (in CUDA memory)
    char *end;  // the end of the block (in CUDA memory)
    SubRegion *subregion;  // 指向此内存块所属的SubRegion的指针.
    bool allocated;  // 如果当前已将此MemoryBlock分配给用户，则为true；否则为false。

    size_t t;        // 如果从未将该存储块提供给用户，则为零；
                     //否则，时间值（CuAllocator类中的t_）为最近被分配给用户或由用户释放的时间。

    std::thread::id thread_id;  // 如果 allocated == false and t > 0 (例如：该内存块由用户释放), 表示释放的线程id

    MemoryBlock *next;  // 此MemoryRegion中的下一个MemoryBlock（如果是最后一个，则为NULL）；
                        //它的“开始”将与该块的“结束”相同。
    MemoryBlock *prev;  // 此MemoryRegion中的前一个MemoryBlock（如果是第一个，则为NULL）； 
                        //它的“结束”将与该块的“开始”相同。

  };

  // MemoryRegion是通过CudaMalloc分配的一大块内存。 通常不会超过3或4。
  //我们将通过size_t（例如0、1、2、3 ...）来标识MemoryRegions，该size_t是memory_regions_vector的索引。
  struct MemoryRegion {
    char *begin;  // 'begin' is the start of the memory region.
    char *end;  // 'end' is the end of the memory region.
    SubRegion *subregion_begin;  // The first SubRegion that belongs to this
                                 // MemoryRegion.
    MemoryBlock *block_begin;  // The first MemoryBlock that belongs to this
                               // MemoryRegion.
  };

  // SubRegion是MemoryRegion中较小的内存区域。 例如，我们将分配的第一个MemoryRegion划分为10个块，如果以后再分配存储块，
  //则将它们细分为大约相同大小的块。 SubRegion只是一个小容器，我们在其中放入了恰好在该SubRegion中开始的所有内存块； 
  //实际上，存储块可能会跨越子区域的边界。 将MemoryRegions划分为SubRegions的动机是，它分配了一种有效的机制，
  //将较小的内存块隔离到较高的内存中，将较大的内存块隔离到较低的内存中：对于每次分配，我们从能够分配某些内容的最高编号的
  //SubRegion中进行分配 那个大小。 随着时间的流逝，这将导致较小的存储块集中在编号较高的子区域中。
  struct SubRegion {
    size_t memory_region;  // memory_regions_ vector索引，用于标识此SubRegion属于哪个MemoryRegion
    size_t subregion_index;  // 该subRegion在subregions_vector内的索引；
                             // 当我们分配更多的MemoryRegions时，这可能会改变。
    char *begin;  // 'begin' is the start of the memory in this SubRegion.
    char *end;    // 'end' is the end of the memory in this SubRegion.

    // 包含在此SubRegion中开始的可用MemoryBlocks.
    std::set<std::pair<size_t, MemoryBlock*> > free_blocks;

    // 指向此MemoryRegion中下一个SubRegion的指针（即，SubRegion，其开始位置等于该位置的结束位置）；
    // 如果为最后一个，则为NULL。
    SubRegion *next;
  };

  // 尝试分配给定大小的CUDA内存； 如果无法分配将崩溃.
  inline void* MallocInternal(size_t size);

  // 在确定可以满足此请求之后，从给定的SubRegion分配。 为了清楚起见，从MallocInternal中分解出来。
  inline void* MallocFromSubregion(SubRegion *subregion, size_t size);


  // 拆分给定的MemoryBlock，
  inline MemoryBlock *SplitBlock(MemoryBlock *block, size_t size);

  // 从其所属的SubRegion的“ free_blocks”集中删除此块。 分配块时以及从其他地方调用此方法。
  void RemoveFromFreeBlocks(MemoryBlock *block);

  // 将此块添加到它所属的SubRegion的“ free_blocks”集中。 在释放块以及从其他地方释放块时，将调用此方法。
  void AddToFreeBlocks(MemoryBlock *block);

  // 分配失败时将调用此函数，我们需要尝试重新手分配更多的内存。 
  //'size'是请求分配失败的内存块的大小-提供该大小是为了确保我们可以分配至少此大小的新区域。
  void AllocateNewRegion(size_t size);

  // 从AllocateNewRegion（）调用，这可以确保对子区域进行所需的排序，并且还重新计算了large_free_block_array。
  void SortSubregions();

  CuAllocatorOptions opts_;

  std::vector<MemoryRegion> memory_regions_;

  std::vector<SubRegion*> subregions_;

  // 对于sub_regions_中的每个SubRegion，此向量为我们提供该SubRegion中存在的最大空闲块的大小，
  //该大小等于sub_regions_ [i]-> free_blocks.begin()-> first。 
  //它使我们能够相当有效地找到可以处理特定内存请求的编号最小的SubRegion。
  std::vector<size_t> largest_free_block_;

  size_t t_;  // 时间计数器，每次调用增加.
  size_t synchronize_gpu_t_;     // 上次调用SynchronizeGpu()时的t_值.
  size_t num_synchronizations_;  // 调用 SynchronizeGpu()次数
  double tot_time_taken_;  // 调用此对象花费的总时间.
  double malloc_time_taken_;  // 在调用cudaMalloc()上的总时间.

  // 这是从用户当前拥有的内存位置到MemoryBlock的映射，后者存储有关该位置的信息。
  std::unordered_map<void*, MemoryBlock*> allocated_block_map_;

  // this is only locked by the '*Locking' versions of the functions (necessary only
  // in multi-threaded applications).
  std::mutex mutex_;

  // 跟踪缓存中的内存使用情况以跟踪应用程序使用的最大内存
  size_t max_allocated_memory_;
  size_t allocated_memory_;
};
```

```cpp
struct CuAllocatorOptions {
  // 如果要在此设备上实际缓存内存分配，则为true。 
  //通常，仅当您想使用cuda-memcheck或cuda-gdb调试可能的内存问题时，才将其设置为false。 
  //它将变慢，但是使用CUDA的本机分配器可以使那些工具检测区域外的内存访问。
  bool cache_memory;

  // CuAllocator分配开始的设备内存的比例； 默认情况下，此值为0.5，如果要共享设备（不建议！），则应将此值设置得较低。
  BaseFloat memory_proportion;

  // 整个CUDA设备内存的目标子区域数;更多区域将使其更积极地整合内存低地址
  int32 num_subregions;

  CuAllocatorOptions():
      cache_memory(true), memory_proportion(0.5), num_subregions(20) { }

  void Register(OptionsItf *po) {
    po->Register("cuda-cache-memory", &cache_memory, "True if you want "
                 "to use the caching allocator.  Set this to false only if you "
                 "want to use cuda-memcheck or cuda-gdb; it will be slower.");
    po->Register("cuda-memory-proportion", &memory_proportion,
                 "Proportion of the GPU device memory that the allocator "
                 "should allocate at the start");
  }

  void Check() {
    // don't let it get too close to 1;
    KALDI_ASSERT(memory_proportion >= 0.05 && memory_proportion < 0.99);
  }
};
```

```cpp
// 获取整个GPU内存使用信息
//  an example showing the format is:
//  "free: 10M, used: 490M, total: 500M: free/total: 0.02"
// In addition, if the pointers 'free' and 'total' are non-NULL, it will
// output to them the free memory and the total memory of the device.
std::string GetFreeGpuMemory(int64* free, int64* total);
```

cudamatrix中关于数组和矩阵的操作和cpu版本类似，详情参考cpu版本说明

