# kaldi源码解析4-matrix

kaldi-matrix中实现了vector和matrix的各种运算，其中矩阵包含compressed-matrix（压缩矩阵），packed-matrix（堆积矩阵），sparse-matrix（稀疏矩阵），sp-matrix（对称矩阵），tp-matrix（三角矩阵）

下面从源码具体来看：

## cblas-wrappers和kaldi-blas

根据kaldi编译选项选择使用哪个加速库

HAVE_ATLAS：有ATLAS，其中包括ATLAS CBLAS实现以及CLAPACK的子集（但函数声明中包含clapack_）

HAVE_CLAPACK：有CBLAS（某些实现）和CLAPACK

HAVE_MKL：使用mkl加速库（当前kaldi默认使用mkl加速）

HAVE_OPENBLAS：使用openblas加速库

## jama-svd和jama-eig

jama: A Java Matrix Package

对jama中奇异值分解和特征分解的C++实现

定义了HAVE_ATLAS编译选项或USE_KALDI_SVD编译选项时才使用

```cpp
#if !defined(HAVE_ATLAS) && !defined(USE_KALDI_SVD)
  // protected:
  // Should be protected but used directly in testing routine.
  // destroys *this!
  void LapackGesvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                     MatrixBase<Real> *Vt);
#else
 protected:
  // destroys *this!
  bool JamaSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
               MatrixBase<Real> *V);

#endif
```

## matrix-lib

用来包含matrix下所有头文件

## matrix-common

matrix-common中定义了矩阵转置类型，resize类型，stride类型和对称矩阵拷贝类型，并定义了举证的下标类型；

同时还声明了各种矩阵的类。

```cpp
typedef enum {
  kTrans    = 112, // = CblasTrans
  kNoTrans  = 111  // = CblasNoTrans
} MatrixTransposeType;
typedef enum {
  kSetZero,      //resize之后把矩阵所有元素都置为0
  kUndefined,    //resize之后随机初始化矩阵
  kCopyData      //resize之后将原有矩阵中的数据拷贝过来，新数据使用0填充
} MatrixResizeType;
typedef enum {
  kDefaultStride,          //stride以字节为单位
  kStrideEqualNumCols,    //stride等于列数
} MatrixStrideType;
typedef enum {
  kTakeLower,
  kTakeUpper,
  kTakeMean,
  kTakeMeanAndCheck
} SpCopyType;
typedef int32 MatrixIndexT;
typedef int32 SignedMatrixIndexT;
typedef uint32 UnsignedMatrixIndexT;
```

## kaldi-vector

### VectorBase

kaldi中vector的基类，封装了基本操作和内存优化

```cpp
void SetZero();  //将vector中的值设置为0
bool IsZero(Real cutoff = 1.0e-06) const;   //判断vector中值是否全为0
void Set(Real f);  //将vector中的所有值设置为指定值
void SetRandn();  //使用随机的正态分布初始化vector
void SetRandUniform();  //设置为均匀分布在（0,1）上的数字
MatrixIndexT RandCategorical() const;  //随机返回一个index
inline MatrixIndexT Dim() const { return dim_; }  //vector的长度
inline MatrixIndexT SizeInBytes() const { return (dim_*sizeof(Real)); } //vector的字节数
inline Real* Data() { return data_; }  //vector数据的起始指针
inline const Real* Data() const { return data_; }  //vector数据的起始指针(const类型)
inline Real operator() (MatrixIndexT i) const {}   //根据索引返回数据，如vector(1)
inline Real & operator() (MatrixIndexT i) {}  //根据索引返回数据，如vector(1)
//返回Vector中部分数据，o表示起始位置，l表示从起始位置开始的长度
SubVector<Real> Range(const MatrixIndexT o, const MatrixIndexT l){}
const SubVector<Real> Range(const MatrixIndexT o,const MatrixIndexT l) const{}
void CopyFromVec(const VectorBase<Real> &v); //从另一个vector中拷贝数据，要求两个vector大小相同
//从SpMatrix或TpMatrix拷贝一个三角的数据到vector，要求一个数据大小相同
template<typename OtherReal>
void CopyFromPacked(const PackedMatrix<OtherReal> &M);
// 从数据类型不同的vector中拷贝数据
template<typename OtherReal>
void CopyFromVec(const VectorBase<OtherReal> &v);
//从CuVector中拷贝数据
template<typename OtherReal>
void CopyFromVec(const CuVectorBase<OtherReal> &v);
// 设置一个最小值来截取数据，数据中所有小于floor_val的数被设置为floor_val，
//floored_count不为空时用来计数被改变数值的个数
void Floor(const VectorBase<Real> &v,Real floor_val,MatrixIndexT *floored_count=nullptr);
inline void ApplyFloor(Real floor_val, MatrixIndexT *floored_count = nullptr);
//对floor_vec执行Floor，返回被改变数值的个数
MatrixIndexT ApplyFloor(const VectorBase<Real> &floor_vec);
// 设置一个最大值来截取数据，数据中所有大于ceil_val的数被设置为ceil_val
//ceiled_count不为空时用来计数被改变数值的个数
void Ceiling(const VectorBase<Real> &v,Real ceil_val,MatrixIndexT *ceiled_count=nullptr);
inline void ApplyCeiling(Real ceil_val, MatrixIndexT *ceiled_count = nullptr);
// 对vector中数据进行多少次方，x^power,如果power为1/2则为开根号,power为负数则再求倒数
void Pow(const VectorBase<Real> &v, Real power);
inline void ApplyPow(Real power);
void ApplyLog();  //对vector中数据进行自然对数运算 Loge
void ApplyLogAndCopy(const VectorBase<Real> &v);  //Loge(v)
void ApplyExp();  //e^x
void ApplyAbs();  //|x|
//将向量的所有元素的绝对值取幂。如果include_sign == true，则输入元素的符号不变，只对数值进行操作。
//如果power为负且输入值为零，则将输出设置为零。
void ApplyPowAbs(Real power, bool include_sign=false);
Real ApplySoftMax();  //\f$ x(i) = exp(x(i)) / \sum_i exp(x(i)) \f$
Real ApplyLogSoftMax();  //\f$ x(i) = x(i) - log(\sum_i exp(x(i))) \f$
void Tanh(const VectorBase<Real> &src);  //对src中元素执行tanh操作赋值到当前vector中
void Sigmoid(const VectorBase<Real> &src);  //对src中元素执行sigmoid操作赋值到当前vector中
Real Norm(Real p) const;  //计算向量的第p范数。
//如果((*this)-other).Norm(2.0) <= tol * (*this).Norm(2.0)则返回true
bool ApproxEqual(const VectorBase<Real> &other, float tol = 0.01) const;
void InvertElements();   //对vector中每个元素求倒数
template<typename OtherReal>
void AddVec(const Real alpha, const VectorBase<OtherReal> &v);//*this = *this +alpha * rv
void AddVec2(const Real alpha, const VectorBase<Real> &v);// *this = *this + alpha * rv^2
template<typename OtherReal>
void AddVec2(const Real alpha, const VectorBase<OtherReal> &v);
//this <-- beta*this + alpha*M*v,Calls BLAS GEMV.
void AddMatVec(const Real alpha, const MatrixBase<Real> &M,
            const MatrixTransposeType trans,  const VectorBase<Real> &v,const Real beta);
//这与AddMatVec相同，但针对v包含很多零的位置进行了优化
void AddMatSvec(const Real alpha, const MatrixBase<Real> &M,
            const MatrixTransposeType trans,  const VectorBase<Real> &v,const Real beta);
//this <-- beta*this + alpha*M*v.   Calls BLAS SPMV.
void AddSpVec(const Real alpha, const SpMatrix<Real> &M,
                const VectorBase<Real> &v, const Real beta);
//this <-- beta*this + alpha*M*v.
void AddTpVec(const Real alpha, const TpMatrix<Real> &M,
            const MatrixTransposeType trans, const VectorBase<Real> &v,const Real beta);
void ReplaceValue(Real orig, Real changed);  //y = (x == orig ? changed : x)
void MulElements(const VectorBase<Real> &v);  //点乘
template<typename OtherReal>
void MulElements(const VectorBase<OtherReal> &v);
void DivElements(const VectorBase<Real> &v);  //点除（对应位置除）
template<typename OtherReal>
void DivElements(const VectorBase<OtherReal> &v);
void Add(Real c);  //vector中每个元素加上c
//this <-- alpha * v .* r + beta*this .
void AddVecVec(Real alpha,const VectorBase<Real> &v,const VectorBase<Real> &r,Real beta);
//this <---- alpha*v/r + beta*this
void AddVecDivVec(Real alpha, const VectorBase<Real> &v,  
                    const VectorBase<Real> &r, Real beta);
void Scale(Real alpha);  //vector每个元素乘以alpha
// 乘以下三角矩阵，*this <-- *this *M
void MulTp(const TpMatrix<Real> &M, const MatrixTransposeType trans); 
//M x = b或M' x = b；b是*this的输入，x是*this的输出
void Solve(const TpMatrix<Real> &M, const MatrixTransposeType trans);
//
void CopyRowsFromMat(const MatrixBase<Real> &M);  //flatten操作，vector长度等于M的行*列
template<typename OtherReal>
void CopyRowsFromMat(const MatrixBase<OtherReal> &M);
void CopyRowsFromMat(const CuMatrixBase<Real> &M);
//逐列进行拼接
void CopyColsFromMat(const MatrixBase<Real> &M);
//拷贝矩阵中的指定行
void CopyRowFromMat(const MatrixBase<Real> &M, MatrixIndexT row);  
template<typename OtherReal>
void CopyRowFromMat(const MatrixBase<OtherReal> &M, MatrixIndexT row);
template<typename OtherReal>
void CopyRowFromSp(const SpMatrix<OtherReal> &S, MatrixIndexT row);
//拷贝矩阵中的某一列
template<typename OtherReal>
void CopyColFromMat(const MatrixBase<OtherReal> &M , MatrixIndexT col);
//提取矩阵的对角线
void CopyDiagFromMat(const MatrixBase<Real> &M);
void CopyDiagFromPacked(const PackedMatrix<Real> &M);  //适用于sp和tp矩阵
Real Max() const;   //获取vector中最大值
Real Max(MatrixIndexT *index) const;  //获取最大值及其下标
Real Min() const;    //获取vector中最小值
Real Min(MatrixIndexT *index) const;  //获取最小值及其下标
Real Sum() const;   //求和
Real SumLog() const;  //先log，再求和
//*this = alpha * (sum of rows of M) + beta * *this
void AddRowSumMat(Real alpha, const MatrixBase<Real> &M, Real beta = 1.0);
//*this = alpha * (sum of columns of M) + beta * *this
void AddColSumMat(Real alpha, const MatrixBase<Real> &M, Real beta = 1.0);
//将矩阵的对角线乘以自身
// *this = diag(M M^T) +  beta * *this (if trans == kNoTrans)
// *this = diag(M^T M) +  beta * *this (if trans == kTrans)
void AddDiagMat2(Real alpha, const MatrixBase<Real> &M,
                   MatrixTransposeType trans = kNoTrans, Real beta = 1.0);
// *this = diag(M N),M N矩阵相乘之后的对角线
void AddDiagMatMat(Real alpha, const MatrixBase<Real> &M, MatrixTransposeType transM,
                     const MatrixBase<Real> &N, MatrixTransposeType transN,
                     Real beta = 1.0);
//返回log(sum(exp()))，如果prune> 0.0，则忽略小于max-prune的项
Real LogSumExp(Real prune = -1.0) const;
// 读写vector
void Read(std::istream &in, bool binary, bool add = false);
void Write(std::ostream &Out, bool binary) const;
```

### Vector

继承自VectorBase，拥有VectorBase中所有方法;

在Vector类中，构造函数通过resize方法将vector和源数据设置为一致，如果vector和源数据大小一致，则复用内存，如果不一致，则在resize中会调用私有的Init方法去申请内存，Init中会调用base-utils中的KALDI_MEMALIGN方法申请内存。（KALDI_MEMALIGN申请的是一块连续内存）

RemoveElement：删除vector中元素，为保证内存连续，会做大量移动，不建议使用

```cpp
template<typename Real>
class Vector: public VectorBase<Real> {
 public:
  /// Constructor that takes no arguments.  Initializes to empty.
  Vector(): VectorBase<Real>() {}
  explicit Vector(const MatrixIndexT s,
                  MatrixResizeType resize_type = kSetZero)
      : VectorBase<Real>() {  Resize(s, resize_type);  }
    
  template<typename OtherReal>
  explicit Vector(const CuVectorBase<OtherReal> &cu);

  /// Copy constructor.  The need for this is controversial.
  Vector(const Vector<Real> &v) : VectorBase<Real>()  { //  (cannot be explicit)
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

  /// Copy-constructor from base-class, needed to copy from SubVector.
  explicit Vector(const VectorBase<Real> &v) : VectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

  /// Type conversion constructor.
  template<typename OtherReal>
  explicit Vector(const VectorBase<OtherReal> &v): VectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

// Took this out since it is unsafe : Arnab
//  /// Constructor from a pointer and a size; copies the data to a location
//  /// it owns.
//  Vector(const Real* Data, const MatrixIndexT s): VectorBase<Real>() {
//    Resize(s);
  //    CopyFromPtr(Data, s);
//  }


  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(Vector<Real> *other);

  /// Destructor.  Deallocates memory.
  ~Vector() { Destroy(); }

  /// Read function using C++ streams.  Can also add to existing contents
  /// of matrix.
  void Read(std::istream &in, bool binary, bool add = false);

  /// Set vector to a specified size (can be zero).
  /// The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  /// This function takes time proportional to the number of data elements.
  void Resize(MatrixIndexT length, MatrixResizeType resize_type = kSetZero);

  /// Remove one element and shifts later elements down.
  void RemoveElement(MatrixIndexT i);

  /// Assignment operator.
  Vector<Real> &operator = (const Vector<Real> &other) {
    Resize(other.Dim(), kUndefined);
    this->CopyFromVec(other);
    return *this;
  }

  /// Assignment operator that takes VectorBase.
  Vector<Real> &operator = (const VectorBase<Real> &other) {
    Resize(other.Dim(), kUndefined);
    this->CopyFromVec(other);
    return *this;
  }
 private:
  /// Init assumes the current contents of the class are invalid (i.e. junk or
  /// has already been freed), and it sets the vector to newly allocated memory
  /// with the specified dimension.  dim == 0 is acceptable.  The memory contents
  /// pointed to by data_ will be undefined.
  void Init(const MatrixIndexT dim);

  /// Destroy function, called internally.
  void Destroy();

};
```

### SubVector

继承自VectorBase，拥有VectorBase所有方法，**SubVector不会去申请内存**，只是操作Vector的部分内存或者Matrix的某一行的内存

```cpp
template<typename Real>
class SubVector : public VectorBase<Real> {
 public:
  SubVector(const VectorBase<Real> &t, const MatrixIndexT origin,
            const MatrixIndexT length) : VectorBase<Real>() {
    // following assert equiv to origin>=0 && length>=0 &&
    // origin+length <= rt.dim_
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(origin)+
                 static_cast<UnsignedMatrixIndexT>(length) <=
                 static_cast<UnsignedMatrixIndexT>(t.Dim()));
    this->data_ = const_cast<Real*> (t.Data()+origin);
    this->dim_   = length;
  }

  /// This constructor initializes the vector to point at the contents
  /// of this packed matrix (SpMatrix or TpMatrix).
  SubVector(const PackedMatrix<Real> &M) {
    this->data_ = const_cast<Real*> (M.Data());
    this->dim_   = (M.NumRows()*(M.NumRows()+1))/2;
  }

  /// Copy constructor
  SubVector(const SubVector &other) : VectorBase<Real> () {
    // this copy constructor needed for Range() to work in base class.
    this->data_ = other.data_;
    this->dim_ = other.dim_;
  }

  /// Constructor from a pointer to memory and a length.  Keeps a pointer
  /// to the data but does not take ownership (will never delete).
  /// Caution: this constructor enables you to evade const constraints.
  SubVector(const Real *data, MatrixIndexT length) : VectorBase<Real> () {
    this->data_ = const_cast<Real*>(data);
    this->dim_   = length;
  }

  /// This operation does not preserve const-ness, so be careful.
  SubVector(const MatrixBase<Real> &matrix, MatrixIndexT row) {
    this->data_ = const_cast<Real*>(matrix.RowData(row));
    this->dim_   = matrix.NumCols();
  }

  ~SubVector() {}  ///< Destructor (does nothing; no pointers are owned here).

 private:
  /// Disallow assignment operator.
  SubVector & operator = (const SubVector &other) {}
};
```

另外kaldi-vector还重载了流运算符，可以直接将数据写入到流或者从流中直接读数据，但是直接写入流的数据是text形式的。

```cpp
template<typename Real>
std::ostream & operator << (std::ostream & out, const VectorBase<Real> & v);
template<typename Real>
std::istream & operator >> (std::istream & in, VectorBase<Real> & v);
template<typename Real>
std::istream & operator >> (std::istream & in, Vector<Real> & v);
```

## kaldi-matrix

-DKALDI_PARANOID通过index访问数据时会检查是否越界

kaldi-matrix还实现了对Htk和Sphinx数据读写，在此不做讨论。

### MatrixBase

```cpp
inline MatrixIndexT  NumRows() const { return num_rows_; } //行数
inline MatrixIndexT NumCols() const { return num_cols_; }  //列数
inline MatrixIndexT Stride() const {  return stride_; }  //行与行之间的间距，>= NumCols
size_t  SizeInBytes() const{} //matrix数据所占字节数
inline const Real* Data() const{return data_;} //返回数据指针
inline Real* Data() { return data_; }
inline  Real* RowData(MatrixIndexT i){} //返回指定行数据
inline const Real* RowData(MatrixIndexT i) const {}
inline Real&  operator() (MatrixIndexT r, MatrixIndexT c) {} //返回索引数据，如M(0,1)
Real &Index (MatrixIndexT r, MatrixIndexT c){}
inline const Real operator() (MatrixIndexT r, MatrixIndexT c) const{}
void SetZero();  //对矩阵置零
void Set(Real);  //使用real设置矩阵
void SetUnit();  //斜对角为1，其余为0（也适用非正方形矩阵）
void SetRandn();  //设置为正态分布的随机值
void SetRandUniform();  //设置为均匀分布在（0，1）上的数字
//从其他矩阵拷贝数据，要确保两个矩阵行数和列数相同
template<typename OtherReal>
void CopyFromMat(const MatrixBase<OtherReal> & M,MatrixTransposeType trans = kNoTrans);
void CopyFromMat(const CompressedMatrix &M);
template<typename OtherReal>
void CopyFromSp(const SpMatrix<OtherReal> &M);
template<typename OtherReal>
void CopyFromTp(const TpMatrix<OtherReal> &M,MatrixTransposeType trans = kNoTrans);
template<typename OtherReal>
void CopyFromMat(const CuMatrixBase<OtherReal> &M,MatrixTransposeType trans = kNoTrans);
//如果v.Dim() == NumRows() *NumCols()，则相当于reshape操作
//如果v.Dim() == NumCols()，则将v中数据赋值给每行
void CopyRowsFromVec(const VectorBase<Real> &v);
void CopyRowsFromVec(const CuVectorBase<Real> &v);
template<typename OtherReal>
void CopyRowsFromVec(const VectorBase<OtherReal> &v);
//如果v.Dim() == NumRows() *NumCols()，则相当于reshape操作
//如果v.Dim() == NumRows()，则将v中数据赋值给每列
void CopyColsFromVec(const VectorBase<Real> &v);
// 将Vector赋值到指定列
void CopyColFromVec(const VectorBase<Real> &v, const MatrixIndexT col);
// 将Vector赋值到指定行
void CopyRowFromVec(const VectorBase<Real> &v, const MatrixIndexT row);
// 将Vector赋值到矩阵的对角线
void CopyDiagFromVec(const VectorBase<Real> &v);
inline const SubVector<Real> Row(MatrixIndexT i) const{} //获取指定行
inline SubVector<Real> Row(MatrixIndexT i){}
//获取子矩阵，row_offset：行起始位置，num_rows子矩阵行数，col_offset列起始位置，num_cols子矩阵列数
inline SubMatrix<Real> Range(const MatrixIndexT row_offset,const MatrixIndexT num_rows,
                       const MatrixIndexT col_offset,const MatrixIndexT num_cols) const{}
inline SubMatrix<Real> RowRange(const MatrixIndexT row_offset,
                                  const MatrixIndexT num_rows) const{}
inline SubMatrix<Real> ColRange(const MatrixIndexT col_offset,
                                  const MatrixIndexT num_cols) const{}
Real Sum() const;  //所有元素和
Real Trace(bool check_square = true) const;  //计算最大正方形子矩阵的对角线元素之和
Real Max() const; //最大值
Real Min() const; //最小值
void MulElements(const MatrixBase<Real> &A);  //点乘
void DivElements(const MatrixBase<Real> &A);  //将每个元素除以给定矩阵的相应元素。
void Scale(Real alpha);  //每个元素乘以alpha
void Max(const MatrixBase<Real> &A);  //将矩阵中的值设置为当前矩阵和A矩阵中对应位置的最大值
void Min(const MatrixBase<Real> &A); //将矩阵中的值设置为当前矩阵和A矩阵中对应位置的最小值
void MulColsVec(const VectorBase<Real> &scale);  //使用vector的值对每列进行缩放
void MulRowsVec(const VectorBase<Real> &scale); //使用vector的值对每行进行缩放
void MulRowsGroupMat(const MatrixBase<Real> &src); //对矩阵进行分组，对每个分组进行scale
Real LogDet(Real *det_sign = NULL) const;  //log(det(M)),计算行列式后再求log
//矩阵求逆，如果log_det不为空，inverse_needed = false，则使用垃圾数据填充矩阵
void Invert(Real *log_det = NULL, Real *det_sign = NULL,bool inverse_needed = true);
void InvertDouble(Real *LogDet = NULL, Real *det_sign = NULL,bool inverse_needed = true);
void InvertElements();   //对每个元素去倒数
void Transpose();  //矩阵转置
//拷贝矩阵的部分列数据，要求src.NumRows()==this.NumRows()，列数可以不相同
void CopyCols(const MatrixBase<Real> &src,const MatrixIndexT *indices);
//拷贝矩阵的部分行数据，要求src.NumCols()==this.NumCols()，行数可以不相同
void CopyRows(const MatrixBase<Real> &src,const MatrixIndexT *indices);
//Add矩阵的部分列数据，要求src.NumRows()==this.NumRows()，列数可以不相同
void AddCols(const MatrixBase<Real> &src,const MatrixIndexT *indices);
void CopyRows(const Real *const *src);  //从二维数组里拷贝数据
void CopyToRows(Real *const *dst) const;  //将矩阵中数据拷贝到二维数组
//this.Row(r) += alpha * src.row(indexes[r]),src.NumCols()==this.NumCols()
void AddRows(Real alpha,const MatrixBase<Real> &src,const MatrixIndexT *indexes);
void AddToRows(Real alpha,const MatrixIndexT *indexes,MatrixBase<Real> *dst) const;
void AddRows(Real alpha, const Real *const *src);  //从二维数组里Add数据
void AddToRows(Real alpha, Real *const *dst) const;  //将矩阵中数据Add到二维数组
inline void ApplyPow(Real power);  //幂次方，1/2表示根号，-1表示倒数
//pow(abs(data[col]), power),include_sign表示是否保留符号
inline void ApplyPowAbs(Real power, bool include_sign=false);
void PowAbs(const MatrixBase<Real> &src, Real power, bool include_sign=false);
bool Power(Real pow);
void Pow(const MatrixBase<Real> &src, Real power);
inline void ApplyHeaviside();//大于零的元素置为1，小于零的元素置为0
void Heaviside(const MatrixBase<Real> &src);
// 设置一个最小值来截取数据，数据中所有小于floor_val的数被设置为floor_val
inline void ApplyFloor(Real floor_val);
void Floor(const MatrixBase<Real> &src, Real floor_val);
// 设置一个最大值来截取数据，数据中所有大于ceiling_val的数被设置为ceiling_val
inline void ApplyCeiling(Real ceiling_val);
void Ceiling(const MatrixBase<Real> &src, Real ceiling_val);
inline void ApplyExp();  //e^x
void Exp(const MatrixBase<Real> &src);
inline void ApplyExpSpecial(); //小于零：e^x;大于零：x+1
void ExpSpecial(const MatrixBase<Real> &src);
//x<lower_limit:e^lower_limit;x>upper_limit:e^upper_limit;否则e^x
inline void ApplyExpLimited(Real lower_limit, Real upper_limit);
void ExpLimited(const MatrixBase<Real> &src, Real lower_limit, Real upper_limit);
inline void ApplyLog(); //Log(x)
void Log(const MatrixBase<Real> &src);
void SoftHinge(const MatrixBase<Real> &src); //y = log(1 + exp(x))
//特征分解
void Eig(MatrixBase<Real> *P,VectorBase<Real> *eigs_real,              
         VectorBase<Real> *eigs_imag) const;
//奇异值分解
void DestructiveSvd(VectorBase<Real> *s, MatrixBase<Real> *U,MatrixBase<Real> *Vt);
void Svd(VectorBase<Real> *s, MatrixBase<Real> *U,MatrixBase<Real> *Vt) const;
void Svd(VectorBase<Real> *s) const { Svd(s, NULL, NULL); }
Real MinSingularValue() const;  //返回最小的奇异值。
void TestUninitialized() const; //如果矩阵是未初始化的内存,则抛出异常
Real Cond() const;  //通过计算SVD返回条件编号
bool IsSymmetric(Real cutoff = 1.0e-05) const; //如果矩阵是对称的，则返回true
bool IsDiagonal(Real cutoff = 1.0e-05) const; //如果矩阵是对角线，则返回true。
bool IsUnit(Real cutoff = 1.0e-05) const;//如果矩阵全为零（对角线上的除外），则返回true。
bool IsZero(Real cutoff = 1.0e-05) const;//如果矩阵全为零则返回true
Real FrobeniusNorm() const;//Frobenius范数，是平方元素之和的平方根。
//如果((*this)-other).FrobeniusNorm() <= tol * (*this).FrobeniusNorm()返回true
bool ApproxEqual(const MatrixBase<Real> &other, float tol = 0.01) const;
bool Equal(const MatrixBase<Real> &other) const;//完全相等。 通常最好使用ApproxEqual。
Real LargestAbsElem() const;  //绝对值的最大值
//返回log(sum(exp()))，如果prune> 0.0，则忽略小于max-prune的项
Real LogSumExp(Real prune = -1.0) const; 
Real ApplySoftMax();  //softmax
void Sigmoid(const MatrixBase<Real> &src);
//y(i) = (sum_{j = i*G}^{(i+1)*G-1} x_j^(power))^(1 / p)
void GroupPnorm(const MatrixBase<Real> &src, Real power);
//计算上述GroupPnorm函数的导数
void GroupPnormDeriv(const MatrixBase<Real> &input, const MatrixBase<Real> &output,
                       Real power);
//y(i) = (max_{j = i*G}^{(i+1)*G-1} x_j
void GroupMax(const MatrixBase<Real> &src);
//GroupMax的导数
void GroupMaxDeriv(const MatrixBase<Real> &input, const MatrixBase<Real> &output);
void Tanh(const MatrixBase<Real> &src);  
//在Sigmoid函数的反向传播导数中使用的函数,*this = diff * value * (1.0 - value).
void DiffSigmoid(const MatrixBase<Real> &value,const MatrixBase<Real> &diff);
//在Tanh函数的反向传播导数中使用的函数,*this = diff * (1.0 - value^2).
void DiffTanh(const MatrixBase<Real> &value,const MatrixBase<Real> &diff);
//使用Svd计算对称正半定矩阵的特征值分解
void SymPosSemiDefEig(VectorBase<Real> *s, MatrixBase<Real> *P,
                        Real check_thresh = 0.001);
void Add(const Real alpha);  //每个元素加alpha
void AddToDiag(const Real alpha);  //对角线元素加alpha
//*this += alpha * a * b^T
template<typename OtherReal>
void AddVecVec(const Real alpha, const VectorBase<OtherReal> &a,
                 const VectorBase<OtherReal> &b);
// [each row of *this] += alpha * v
template<typename OtherReal>
void AddVecToRows(const Real alpha, const VectorBase<OtherReal> &v);
// [each col of *this] += alpha * v
template<typename OtherReal>
void AddVecToCols(const Real alpha, const VectorBase<OtherReal> &v);
// *this += alpha * M [or M^T]
void AddMat(const Real alpha, const MatrixBase<Real> &M,
              MatrixTransposeType transA = kNoTrans);
void AddSmat(Real alpha, const SparseMatrix<Real> &A,
               MatrixTransposeType trans = kNoTrans);
//(*this) = alpha * op(A) * B + beta * (*this)
void AddSmatMat(Real alpha, const SparseMatrix<Real> &A,
                  MatrixTransposeType transA, const MatrixBase<Real> &B,Real beta);
//(*this) = alpha * A * op(B) + beta * (*this)
void AddMatSmat(Real alpha, const MatrixBase<Real> &A,
                  const SparseMatrix<Real> &B, MatrixTransposeType transB,Real beta);
//对称矩阵操作，仅更新下半部分，*this = beta * *this + alpha * M M^T
void SymAddMat2(const Real alpha, const MatrixBase<Real> &M,
                  MatrixTransposeType transA, Real beta);
//与add M相同，但将每行M_i缩放v（i）,*this = beta * *this + alpha * diag(v) * M [or M^T].
void AddDiagVecMat(const Real alpha, const VectorBase<Real> &v,
                 const MatrixBase<Real> &M, MatrixTransposeType transM,Real beta = 1.0);
//与add M相同，但将每列M_j缩放v（j）。*this = beta * *this + alpha * M [or M^T] * diag(v)
void AddMatDiagVec(const Real alpha,const MatrixBase<Real> &M, 
                   MatrixTransposeType transM,VectorBase<Real> &v,Real beta = 1.0);
//*this = beta * *this + alpha * A .* B (.* 表示点乘)
void AddMatMatElements(const Real alpha,const MatrixBase<Real>& A,
                         const MatrixBase<Real>& B,const Real beta);
template<typename OtherReal>
void AddSp(const Real alpha, const SpMatrix<OtherReal> &S);  //*this += alpha * S
//*this = beta * *this + alpha * A * B
void AddMatMat(const Real alpha,const MatrixBase<Real>& A, MatrixTransposeType transA,
                 const MatrixBase<Real>& B, MatrixTransposeType transB,const Real beta);
void AddMatSmat(const Real alpha,const MatrixBase<Real>& A, MatrixTransposeType transA,
                  const MatrixBase<Real>& B, MatrixTransposeType transB,const Real beta);
void AddSmatMat(const Real alpha,const MatrixBase<Real>& A, MatrixTransposeType transA,
                  const MatrixBase<Real>& B, MatrixTransposeType transB,const Real beta);
void AddSpMat(const Real alpha,const SpMatrix<Real>& A,
                const MatrixBase<Real>& B, MatrixTransposeType transB,const Real beta);
void AddTpMat(const Real alpha,const TpMatrix<Real>& A, MatrixTransposeType transA,
                const MatrixBase<Real>& B, MatrixTransposeType transB,const Real beta);
void AddMatSp(const Real alpha,const MatrixBase<Real>& A, MatrixTransposeType transA,
                const SpMatrix<Real>& B,const Real beta);
void AddMatTp(const Real alpha,const MatrixBase<Real>& A, MatrixTransposeType transA,
                const TpMatrix<Real>& B, MatrixTransposeType transB,const Real beta);
void AddTpTp(const Real alpha,const TpMatrix<Real>& A, MatrixTransposeType transA,
               const TpMatrix<Real>& B, MatrixTransposeType transB,const Real beta);
void AddSpSp(const Real alpha,const SpMatrix<Real>& A, const SpMatrix<Real>& B,
               const Real beta);
//*this = a * b / c (by element; when c = 0, *this = a)
void SetMatMatDivMat(const MatrixBase<Real>& A,
                       const MatrixBase<Real>& B,const MatrixBase<Real>& C);
//this <-- beta*this + alpha*A*B*C.
void AddMatMatMat(const Real alpha,const MatrixBase<Real>& A, MatrixTransposeType transA,
                    const MatrixBase<Real>& B, MatrixTransposeType transB,
                  const MatrixBase<Real>& C, MatrixTransposeType transC,const Real beta);
void AddSpMatSp(const Real alpha,const SpMatrix<Real> &A,const MatrixBase<Real>& B, 
                MatrixTransposeType transB,const SpMatrix<Real>& C,const Real beta);
void CopyLowerToUpper();  //将下三角复制到上三角（对称）
void CopyUpperToLower();  //将上三角复制到下三角（对称）
void OrthogonalizeRows();  //此函数使用Gram-Schmidt过程正交化矩阵的行
//奇异值分解
void LapackGesvd(VectorBase<Real> *s, MatrixBase<Real> *U,MatrixBase<Real> *Vt);
// 读写matrix
void Read(std::istream & in, bool binary, bool add = false);
void Write(std::ostream & out, bool binary) const;
```

### Matrix

继承MatrixBase，有用MatrixBase所有方法，另外还定义了一些特有方法

在Matrix类中，构造函数通过Resize方法将Matrix和源数据设置为一致，如果Matrix和源数据大小一致，则复用内存，如果不一致，则在Resize中会调用私有的Init方法去申请内存，Init中会调用base-utils中的KALDI_MEMALIGN方法申请内存。（KALDI_MEMALIGN申请的是一块连续内存）

RemoveRow：删除Matrix中一行，为保证内存连续，会做大量移动，不建议使用

Transpose：矩阵转置

```cpp
class Matrix : public MatrixBase<Real> {
 public:

  /// Empty constructor.
  Matrix();

  /// Basic constructor.
  Matrix(const MatrixIndexT r, const MatrixIndexT c,
         MatrixResizeType resize_type = kSetZero,
         MatrixStrideType stride_type = kDefaultStride):
      MatrixBase<Real>() { Resize(r, c, resize_type, stride_type); }
  template<typename OtherReal>
  explicit Matrix(const CuMatrixBase<OtherReal> &cu,
                  MatrixTransposeType trans = kNoTrans);


  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(Matrix<Real> *other);
  void Swap(CuMatrix<Real> *mat);

  /// Constructor from any MatrixBase. Can also copy with transpose.
  /// Allocates new memory.
  explicit Matrix(const MatrixBase<Real> & M,
                  MatrixTransposeType trans = kNoTrans);
  Matrix(const Matrix<Real> & M);  //  (cannot make explicit)
  template<typename OtherReal>
  explicit Matrix(const MatrixBase<OtherReal> & M,
                    MatrixTransposeType trans = kNoTrans);
  template<typename OtherReal>
  explicit Matrix(const SpMatrix<OtherReal> & M) : MatrixBase<Real>() {
    Resize(M.NumRows(), M.NumRows(), kUndefined);
    this->CopyFromSp(M);
  }
  explicit Matrix(const CompressedMatrix &C);
  template <typename OtherReal>
  explicit Matrix(const TpMatrix<OtherReal> & M,
                  MatrixTransposeType trans = kNoTrans) : MatrixBase<Real>() {
    if (trans == kNoTrans) {
      Resize(M.NumRows(), M.NumCols(), kUndefined);
      this->CopyFromTp(M);
    } else {
      Resize(M.NumCols(), M.NumRows(), kUndefined);
      this->CopyFromTp(M, kTrans);
    }
  }
  /// read from stream.
  // Unlike one in base, allows resizing.
  void Read(std::istream & in, bool binary, bool add = false);
  void RemoveRow(MatrixIndexT i);
  void Transpose();
  ~Matrix() { Destroy(); }

  /// Sets matrix to a specified size (zero is OK as long as both r and c are
  /// zero).  The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  ///
  /// You can set stride_type to kStrideEqualNumCols to force the stride
  /// to equal the number of columns; by default it is set so that the stride
  /// in bytes is a multiple of 16.
  ///
  /// This function takes time proportional to the number of data elements.
  void Resize(const MatrixIndexT r,
              const MatrixIndexT c,
              MatrixResizeType resize_type = kSetZero,
              MatrixStrideType stride_type = kDefaultStride);
  /// Assignment operator that takes MatrixBase.
  Matrix<Real> &operator = (const MatrixBase<Real> &other) {
    if (MatrixBase<Real>::NumRows() != other.NumRows() ||
        MatrixBase<Real>::NumCols() != other.NumCols())
      Resize(other.NumRows(), other.NumCols(), kUndefined);
    MatrixBase<Real>::CopyFromMat(other);
    return *this;
  }
  /// Assignment operator. Needed for inclusion in std::vector.
  Matrix<Real> &operator = (const Matrix<Real> &other) {
    if (MatrixBase<Real>::NumRows() != other.NumRows() ||
        MatrixBase<Real>::NumCols() != other.NumCols())
      Resize(other.NumRows(), other.NumCols(), kUndefined);
    MatrixBase<Real>::CopyFromMat(other);
    return *this;
  }
 private:
  void Destroy();
  void Init(const MatrixIndexT r,
            const MatrixIndexT c,
            const MatrixStrideType stride_type);

};
```

### SubMatrix

继承自MatrixBase，拥有MatrixBase所有方法，**SubMatrix不会去申请内存**，只是操作Matrix的部分内存

```cpp
template<typename Real>
class SubMatrix : public MatrixBase<Real> {
 public:
  // Initialize a SubMatrix from part of a matrix; this is
  // a bit like A(b:c, d:e) in Matlab.
  // This initializer is against the proper semantics of "const", since
  // SubMatrix can change its contents.  It would be hard to implement
  // a "const-safe" version of this class.
  SubMatrix(const MatrixBase<Real>& T,
            const MatrixIndexT ro,  // row offset, 0 < ro < NumRows()
            const MatrixIndexT r,   // number of rows, r > 0
            const MatrixIndexT co,  // column offset, 0 < co < NumCols()
            const MatrixIndexT c);   // number of columns, c > 0

  // This initializer is mostly intended for use in CuMatrix and related
  // classes.  Be careful!
  SubMatrix(Real *data,
            MatrixIndexT num_rows,
            MatrixIndexT num_cols,
            MatrixIndexT stride);

  ~SubMatrix<Real>() {}

  /// This type of constructor is needed for Range() to work [in Matrix base
  /// class]. Cannot make it explicit.
  SubMatrix<Real> (const SubMatrix &other):
  MatrixBase<Real> (other.data_, other.num_cols_, other.num_rows_,
                    other.stride_) {}

 private:
  /// Disallow assignment.
  SubMatrix<Real> &operator = (const SubMatrix<Real> &other);
};
```

另外kaldi-matrix还重载了流运算符，可以直接将数据写入到流或者从流中直接读数据，但是直接写入流的数据是text形式的。

```cpp
template<typename Real>
std::ostream & operator << (std::ostream & Out, const MatrixBase<Real> & M);
template<typename Real>
std::istream & operator >> (std::istream & In, MatrixBase<Real> & M);
template<typename Real>
std::istream & operator >> (std::istream & In, Matrix<Real> & M);
```

## compressed-matrix

压缩矩阵，将原矩阵压缩为16位8位矩阵

```cpp
enum CompressionMethod {
  //当您不指定压缩方法时，这是默认设置。 如果行数大于8，则是使用kSpeechFeature的简写，
  //否则，则是kTwoByteAuto。
  kAutomaticMethod = 1,  
  //这是最复杂的压缩方法，专为语音特征而设计，这些语音特征具有大致高斯分布，每个维度的范围都不同。 
  //每个元素都存储在一个字节中，但是每列有一个8字节的头。 整数值的间距不均匀，但在3个范围内。
  kSpeechFeature = 2,
  //每个元素都以uint16的形式存储在两个字节中，并且以矩阵的最小和最大元素为边缘自动选择可表示的值范围。
  kTwoByteAuto = 3,
  //每个元素以uint16的形式存储在两个字节中，其值的可表示范围选择为与存储
  //带符号整数（即[-32768.0、32767.0]）时所得到的值一致。 适用于以前存储为16位PCM的波形数据。
  kTwoByteSignedInteger = 4,
  //每个元素都以uint8的形式存储在一个字节中，并且以矩阵的最小和最大元素为边缘自动选择值的可表示范围。
  kOneByteAuto = 5,
  //每个元素以uint8的形式存储在一个字节中，其可表示的值范围等于[0.0，255.0]。
  kOneByteUnsignedInteger = 6,
  //每个元素都以uint8的形式存储在一个字节中，其可表示的值范围等于[0.0，1.0]。 
  //适用于先前已压缩为int8的图像数据。
  kOneByteZeroOne = 7
};
```

```cpp
class CompressedMatrix {
 public:
  CompressedMatrix(): data_(NULL) { }

  ~CompressedMatrix() { Clear(); }

  template<typename Real>
  explicit CompressedMatrix(const MatrixBase<Real> &mat,
                            CompressionMethod method = kAutomaticMethod):
      data_(NULL) { CopyFromMat(mat, method); }
  //获取压缩矩阵的子矩阵
  CompressedMatrix(const CompressedMatrix &mat,
                   const MatrixIndexT row_offset,
                   const MatrixIndexT num_rows,
                   const MatrixIndexT col_offset,
                   const MatrixIndexT num_cols,
                   bool allow_padding = false);

  void *Data() const { return this->data_; }

  /// 对矩阵使用CompressionMethod方法进行压缩
  template<typename Real>
  void CopyFromMat(const MatrixBase<Real> &mat,
                   CompressionMethod method = kAutomaticMethod);
  CompressedMatrix(const CompressedMatrix &mat);
  CompressedMatrix &operator = (const CompressedMatrix &mat); // assignment operator.
  template<typename Real>
  CompressedMatrix &operator = (const MatrixBase<Real> &mat); // assignment operator.

  //将压缩矩阵复制到一般矩阵，要求两个矩阵大小相同
  template<typename Real>
  void CopyToMat(MatrixBase<Real> *mat,
                 MatrixTransposeType trans = kNoTrans) const;
  //文件读写
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  /// 矩阵的行数和列数
  inline MatrixIndexT NumRows() const { return (data_ == NULL) ? 0 :
      (*reinterpret_cast<GlobalHeader*>(data_)).num_rows; }
  inline MatrixIndexT NumCols() const { return (data_ == NULL) ? 0 :
      (*reinterpret_cast<GlobalHeader*>(data_)).num_cols; }

  /// 将指定行拷贝到vector
  template<typename Real>
  void CopyRowToVec(MatrixIndexT row, VectorBase<Real> *v) const;
  // 将指定列拷贝到vector
  template<typename Real>
  void CopyColToVec(MatrixIndexT col, VectorBase<Real> *v) const;

  /// 将压缩矩阵的子矩阵拷贝到一般矩阵
  template<typename Real>
  void CopyToMat(int32 row_offset,
                 int32 column_offset,
                 MatrixBase<Real> *dest) const;
  void Swap(CompressedMatrix *other) { std::swap(data_, other->data_); }
  void Clear();

  /// 按alpha缩放矩阵的所有元素。
  /// 它按alpha缩放GlobalHeader中的浮点值。
  void Scale(float alpha);

  friend class Matrix<float>;
  friend class Matrix<double>;
 private:
  enum DataFormat {
    kOneByteWithColHeaders = 1,
    kTwoByte = 2,
    kOneByte = 3
  };
  //保留原矩阵信息的结构体
  struct GlobalHeader {
    int32 format;     // Represents the enum DataFormat.
    float min_value;  
    float range;
    int32 num_rows;
    int32 num_cols;
  };
  //kSpeechFeature使用
  struct PerColHeader {
    uint16 percentile_0;
    uint16 percentile_25;
    uint16 percentile_75;
    uint16 percentile_100;
  };
  void *data_; 
}
```

## packed-matrix

PackedMatrix矩阵是SpMatrix（对称矩阵）和TpMatrix（三角矩阵）的基类

在PackedMatrix类中，构造函数通过Resize方法将PackedMatrix和源数据设置为一致，如果PackedMatrix和源数据大小一致，则复用内存，如果不一致，则在Resize中会调用私有的Init方法去申请内存，Init中会调用base-utils中的KALDI_MEMALIGN方法申请内存。（KALDI_MEMALIGN申请的是一块连续内存）

在PackedMatrix中定义了如下方法：

```cpp
void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero);
void AddToDiag(const Real r);  //给对角线元素加r
void ScaleDiag(const Real alpha);  //对角线元素乘alpha
void SetDiag(const Real alpha);   //对角线元素设置为alpha
template<typename OtherReal>
void CopyFromPacked(const PackedMatrix<OtherReal> &orig);
//从vector中拷贝数据，要求vector中数据个数为一个三角数据个数
//orig.Dim() == (NumRows()*(NumRows()+1)) / 2;
template<typename OtherReal>
void CopyFromVec(const SubVector<OtherReal> &orig);
Real* Data() { return data_; }
const Real* Data() const { return data_; }
inline MatrixIndexT NumRows() const { return num_rows_; }
inline MatrixIndexT NumCols() const { return num_rows_; }
size_t SizeInBytes() const{}
Real operator() (MatrixIndexT r, MatrixIndexT c) const; //根据索引访问数据
Real &operator() (MatrixIndexT r, MatrixIndexT c);
Real Max() const;
Real Min() const;
void Scale(Real c);
friend std::ostream & operator << <> (std::ostream & out,const PackedMatrix<Real> &m);
void Read(std::istream &in, bool binary, bool add = false);
void Write(std::ostream &out, bool binary) const;
```

另外还重载了流运算符，可以直接将矩阵写入到流

```cpp
template<typename Real>
std::ostream & operator << (std::ostream & os, const PackedMatrix<Real>& M) {
  M.Write(os, false);
  return os;
}
template<typename Real>
std::istream & operator >> (std::istream &is, PackedMatrix<Real> &M) {
  M.Read(is, false);
  return is;
}
```

## sp-matrix

SpMatrix：对称矩阵

构造函数：

```cpp
explicit SpMatrix(const CuSpMatrix<Real> &cu);
explicit SpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
      : PackedMatrix<Real>(r, resize_type) {}
SpMatrix(const SpMatrix<Real> &orig): PackedMatrix<Real>(orig) {}
template<typename OtherReal>
explicit SpMatrix(const SpMatrix<OtherReal> &orig): PackedMatrix<Real>(orig) {}
explicit SpMatrix(const MatrixBase<Real> & orig,SpCopyType copy_type = kTakeMean);
```

功能函数

```cpp
inline void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero);
void CopyFromSp(const SpMatrix<Real> &other);
template<typename OtherReal>
void CopyFromSp(const SpMatrix<OtherReal> &other);
void CopyFromMat(const MatrixBase<Real> &orig,SpCopyType copy_type = kTakeMean);
inline Real operator() (MatrixIndexT r, MatrixIndexT c) const; //索引
inline Real &operator() (MatrixIndexT r, MatrixIndexT c);
SpMatrix<Real>& operator=(const SpMatrix<Real> &other);
void Invert(Real *logdet = NULL, Real *det_sign= NULL,bool inverse_needed = true);//求逆
void InvertDouble(Real *logdet = NULL, Real *det_sign = NULL,bool inverse_needed = true);
inline Real Cond() const {}; //返回奇异值的最大比率。
void ApplyPow(Real exponent);  //通过Svd将矩阵取小数幂
//对称正定矩阵实现的SVD版本
void SymPosSemiDefEig(VectorBase<Real> *s, MatrixBase<Real> *P,
                        Real tolerance = 0.001) const;
void Eig(VectorBase<Real> *s, MatrixBase<Real> *P = NULL) const;  //特征分解
//对称矩阵的最大特征值和相应的特征向量。
void TopEigs(VectorBase<Real> *s, MatrixBase<Real> *P,
               MatrixIndexT lanczos_dim = 0) const;
Real MaxAbsEig() const;  //特征值绝对值的最大值
bool IsPosDef() const;  //如果Cholesky成功，则返回true。
void AddSp(const Real alpha, const SpMatrix<Real> &Ma);  //*this += alpha * Ma
Real LogPosDefDet() const;//计算对数行列式，但仅适用于+ ve-def矩阵（它使用Cholesky）。
Real LogDet(Real *det_sign = NULL) const;  //Log(Det(x))
template<typename OtherReal>
void AddVec2(const Real alpha, const VectorBase<OtherReal> &v); //this<--this +alpha v v'
//this <-- this + alpha (v w' + w v')
void AddVecVec(const Real alpha, const VectorBase<Real> &v,
                 const VectorBase<Real> &w);
//*this = beta * *thi + alpha * diag(v) * S * diag(v)
void AddVec2Sp(const Real alpha, const VectorBase<Real> &v,
                 const SpMatrix<Real> &S, const Real beta);
//对角线更新，this <-- this + diag(v)
template<typename OtherReal>
void AddDiagVec(const Real alpha, const VectorBase<OtherReal> &v);
//非转置：(*this)=beta*(*this)+alpha * M * M^T,转置(*this)=beta*(*this)+ alpha * M^T * M
void AddMat2(const Real alpha, const MatrixBase<Real> &M,
               MatrixTransposeType transM, const Real beta);
//this <-- beta*this  +  alpha * M * A * M^T.
void AddMat2Sp(const Real alpha, const MatrixBase<Real> &M,
            MatrixTransposeType transM, const SpMatrix<Real> &A,const Real beta = 0.0);
//这是AddMat2Sp的版本，专门用于M相当稀疏的情况，这是使raw-fMLLR代码高效的必需条件。
void AddSmat2Sp(const Real alpha, const MatrixBase<Real> &M,
             MatrixTransposeType transM, const SpMatrix<Real> &A,const Real beta = 0.0);
//this <-- beta*this  +  alpha * T * A * T^T.
void AddTp2Sp(const Real alpha, const TpMatrix<Real> &T,
             MatrixTransposeType transM, const SpMatrix<Real> &A,const Real beta = 0.0);
//this <-- beta*this  +  alpha * T * T^T.
void AddTp2(const Real alpha, const TpMatrix<Real> &T,
              MatrixTransposeType transM, const Real beta = 0.0);
//this <-- beta*this + alpha * M * diag(v) * M^T.
void AddMat2Vec(const Real alpha, const MatrixBase<Real> &M,
            MatrixTransposeType transM, const VectorBase<Real> &v,const Real beta = 0.0);
//this = alpha * Floor,要求Floor为正定矩阵
int ApplyFloor(const SpMatrix<Real> &Floor, Real alpha = 1.0,bool verbose = false);
int ApplyFloor(Real floor);//下限：给定正半定矩阵，将特征值下限到指定数
bool IsDiagonal(Real cutoff = 1.0e-05) const;
bool IsUnit(Real cutoff = 1.0e-05) const;  
bool IsZero(Real cutoff = 1.0e-05) const;
bool IsTridiagonal(Real cutoff = 1.0e-05) const;
Real FrobeniusNorm() const;  //平方和
bool ApproxEqual(const SpMatrix<Real> &other, float tol = 0.01) const;
//通过将所有特征值下限为一个正数来限制对称正半定矩阵的条件为指定值，
//该正数是最大数的正数（如果没有正特征值则为零）
MatrixIndexT LimitCond(Real maxCond = 1.0e+5, bool invert = false);
MatrixIndexT LimitCondDouble(Real maxCond = 1.0e+5, bool invert = false);
void Tridiagonalize(MatrixBase<Real> *Q);  //用正交变换对矩阵进行对角线化。
void Qr(MatrixBase<Real> *Q);  //Qr分解
```

## tp-matrix

TpMatrix：三角矩阵

构造函数：

```cpp
explicit TpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
      : PackedMatrix<Real>(r, resize_type) {}
TpMatrix(const TpMatrix<Real>& orig) : PackedMatrix<Real>(orig) {}
explicit TpMatrix(const CuTpMatrix<Real> &cu);
template<typename OtherReal> 
explicit TpMatrix(const TpMatrix<OtherReal>& orig): PackedMatrix<Real>(orig) {}
```

功能函数

```cpp
Real operator() (MatrixIndexT r, MatrixIndexT c) const; //索引
Real &operator() (MatrixIndexT r, MatrixIndexT c);
void Cholesky(const SpMatrix<Real>& orig);  //Cholesky分解
void Invert();  //求逆
void InvertDouble(); 
void Swap(TpMatrix<Real> *other);
Real Determinant();  //返回矩阵的行列式（对角线的乘积）
void CopyFromMat(const MatrixBase<Real> &M,MatrixTransposeType Trans = kNoTrans);
void CopyFromMat(const CuTpMatrix<Real> &other);
void CopyFromTp(const TpMatrix<Real> &other);
template<typename OtherReal> void CopyFromTp(const TpMatrix<OtherReal> &other);
void AddTp(const Real alpha, const TpMatrix<Real> &M); //*this += alpha * M.
TpMatrix<Real>& operator=(const TpMatrix<Real> &other);
void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero);
```

## sparse-matrix

sparse-matrix是kaldi中定义的数据不连续的vector和matrix类型，其中定义了SparseVector ，SparseMatrix和GeneralMatrix三个类。

SparseVector：数据类型为std::vector<std::pair<MatrixIndexT, Real> > pairs_; MatrixIndexT为数据下标，Real为实际数据。

SparseMatrix：数据结构为std::vector<SparseVector<Real> > rows_; 

GeneralMatrix：能够以以下三种形式之一存储矩阵：Matrix或CompressedMatrix或SparseMatrix。处理读写单个对象类型。 它对于稀疏与否，压缩与否的神经网络训练目标很有用。

### SparseVector

```cpp
class SparseVector {
 public:
  MatrixIndexT Dim() const { return dim_; }
  Real Sum() const;
  template <class OtherReal>
  void CopyElementsToVec(VectorBase<OtherReal> *vec) const;
  // *vec += alpha * *this.
  template <class OtherReal>
  void AddToVec(Real alpha,VectorBase<OtherReal> *vec) const;
  template <class OtherReal>
  void CopyFromSvec(const SparseVector<OtherReal> &other);
  SparseVector<Real> &operator = (const SparseVector<Real> &other);
  SparseVector(const SparseVector<Real> &other) { *this = other; }
  void Swap(SparseVector<Real> *other);
  Real Max(int32 *index) const;  //最大值
  /// 返回非0元素个数
  MatrixIndexT NumElements() const { return pairs_.size(); }
  const std::pair<MatrixIndexT, Real> &GetElement(MatrixIndexT i) const {
    return pairs_[i];
  }
  std::pair<MatrixIndexT, Real> *Data();
  const std::pair<MatrixIndexT, Real> *Data() const;
  void SetRandn(BaseFloat zero_prob);
  SparseVector(): dim_(0) { }
  explicit SparseVector(MatrixIndexT dim): dim_(dim) { KALDI_ASSERT(dim >= 0); }
  SparseVector(MatrixIndexT dim,
               const std::vector<std::pair<MatrixIndexT, Real> > &pairs);
  explicit SparseVector(const VectorBase<Real> &vec);
  void Resize(MatrixIndexT dim, MatrixResizeType resize_type = kSetZero);
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &os, bool binary);
  void Scale(Real alpha);  //缩放
 private:
  MatrixIndexT dim_;
  std::vector<std::pair<MatrixIndexT, Real> > pairs_;
};
```

### SparseMatrix

```cpp
template <typename Real>
class SparseMatrix {
 public:
  MatrixIndexT NumRows() const;
  MatrixIndexT NumCols() const;
  MatrixIndexT NumElements() const;
  Real Sum() const;
  Real FrobeniusNorm() const;  //平方和
  explicit SparseMatrix(const MatrixBase<Real> &mat);
  template <class OtherReal>
  void CopyToMat(MatrixBase<OtherReal> *other,MatrixTransposeType t = kNoTrans) const;
  void CopyElementsToVec(VectorBase<Real> *other) const;
  template<class OtherReal>
  void CopyFromSmat(const SparseMatrix<OtherReal> &other,
                    MatrixTransposeType trans = kNoTrans);
  /// Does *other = *other + alpha * *this.
  void AddToMat(BaseFloat alpha, MatrixBase<Real> *other,
                MatrixTransposeType t = kNoTrans) const;
  SparseMatrix<Real> &operator = (const SparseMatrix<Real> &other);
  SparseMatrix(const SparseMatrix<Real> &other, MatrixTransposeType trans =kNoTrans) {
    this->CopyFromSmat(other, trans);
  }
  void Swap(SparseMatrix<Real> *other);
  SparseVector<Real> *Data();
  const SparseVector<Real> *Data() const;
  SparseMatrix(
      int32 dim,
      const std::vector<std::vector<std::pair<MatrixIndexT, Real> > > &pairs);
  void SetRandn(BaseFloat zero_prob);  //随机初始化
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &os, bool binary);
  const SparseVector<Real> &Row(MatrixIndexT r) const;
  void SetRow(int32 r, const SparseVector<Real> &vec);
  void SelectRows(const std::vector<int32> &row_indexes,
                  const SparseMatrix<Real> &smat_other);
  void AppendSparseMatrixRows(std::vector<SparseMatrix<Real> > *inputs);
  SparseMatrix() { }
  SparseMatrix(int32 num_rows, int32 num_cols) { Resize(num_rows, num_cols); }
  SparseMatrix(const std::vector<int32> &indexes, int32 dim,
               MatrixTransposeType trans = kNoTrans);
  SparseMatrix(const std::vector<int32> &indexes,
               const VectorBase<Real> &weights, int32 dim,
               MatrixTransposeType trans = kNoTrans);
  void Resize(MatrixIndexT rows, MatrixIndexT cols,
              MatrixResizeType resize_type = kSetZero);
  void Scale(Real alpha);  //缩放
 private:
  std::vector<SparseVector<Real> > rows_;
};
```

### GeneralMatrix

```cpp
class GeneralMatrix {
 public:
  /// Returns the type of the matrix: kSparseMatrix, kCompressedMatrix or
  /// kFullMatrix.  If this matrix is empty, returns kFullMatrix.
  GeneralMatrixType Type() const;
  void Compress();  // If it was a full matrix, compresses, changing Type() to
                    // kCompressedMatrix; otherwise does nothing.
  void Uncompress();  // If it was a compressed matrix, uncompresses, changing
                      // Type() to kFullMatrix; otherwise does nothing.
  void Write(std::ostream &os, bool binary) const;
  /// Note: if you write a compressed matrix in text form, it will be read as
  /// a regular full matrix.
  void Read(std::istream &is, bool binary);
  /// Returns the contents as a SparseMatrix.  This will only work if
  /// Type() returns kSparseMatrix, or NumRows() == 0; otherwise it will crash.
  const SparseMatrix<BaseFloat> &GetSparseMatrix() const;
  /// Swaps the with the given SparseMatrix.  This will only work if
  /// Type() returns kSparseMatrix, or NumRows() == 0.
  void SwapSparseMatrix(SparseMatrix<BaseFloat> *smat);
  /// Returns the contents as a compressed matrix.  This will only work if
  /// Type() returns kCompressedMatrix, or NumRows() == 0; otherwise it will
  /// crash.
  const CompressedMatrix &GetCompressedMatrix() const;
  /// Swaps the with the given CompressedMatrix.  This will only work if
  /// Type() returns kCompressedMatrix, or NumRows() == 0.
  void SwapCompressedMatrix(CompressedMatrix *cmat);
  /// Returns the contents as a Matrix<BaseFloat>.  This will only work if
  /// Type() returns kFullMatrix, or NumRows() == 0; otherwise it will crash.
  const Matrix<BaseFloat>& GetFullMatrix() const;
  /// Outputs the contents as a matrix.  This will work regardless of
  /// Type().  Sizes its output, unlike CopyToMat().
  void GetMatrix(Matrix<BaseFloat> *mat) const;
  /// Swaps the with the given Matrix.  This will only work if
  /// Type() returns kFullMatrix, or NumRows() == 0.
  void SwapFullMatrix(Matrix<BaseFloat> *mat);
  /// Copies contents, regardless of type, to "mat", which must be correctly
  /// sized.  See also GetMatrix(), which will size its output for you.
  void CopyToMat(MatrixBase<BaseFloat> *mat,
                 MatrixTransposeType trans = kNoTrans) const;
  /// Copies contents, regardless of type, to "cu_mat", which must be
  /// correctly sized.  Implemented in ../cudamatrix/cu-sparse-matrix.cc
  void CopyToMat(CuMatrixBase<BaseFloat> *cu_mat,
                 MatrixTransposeType trans = kNoTrans) const;
  /// Adds alpha times *this to mat.
  void AddToMat(BaseFloat alpha, MatrixBase<BaseFloat> *mat,
                MatrixTransposeType trans = kNoTrans) const;
  /// Adds alpha times *this to cu_mat.
  /// Implemented in ../cudamatrix/cu-sparse-matrix.cc
  void AddToMat(BaseFloat alpha, CuMatrixBase<BaseFloat> *cu_mat,
                MatrixTransposeType trans = kNoTrans) const;
  /// Scale each element of matrix by alpha.
  void Scale(BaseFloat alpha);
  /// Assignment from regular matrix.
  GeneralMatrix &operator= (const MatrixBase<BaseFloat> &mat);
  /// Assignment from compressed matrix.
  GeneralMatrix &operator= (const CompressedMatrix &mat);
  /// Assignment from SparseMatrix<BaseFloat>
  GeneralMatrix &operator= (const SparseMatrix<BaseFloat> &smat);
  MatrixIndexT NumRows() const;
  MatrixIndexT NumCols() const;
  explicit GeneralMatrix(const MatrixBase<BaseFloat> &mat) { *this = mat; }
  explicit GeneralMatrix(const CompressedMatrix &cmat) { *this = cmat; }
  explicit GeneralMatrix(const SparseMatrix<BaseFloat> &smat) { *this = smat; }
  GeneralMatrix() { }
  // Assignment operator.
  GeneralMatrix &operator =(const GeneralMatrix &other);
  // Copy constructor
  GeneralMatrix(const GeneralMatrix &other) { *this = other; }
  // Sets to the empty matrix.
  void Clear();
  // shallow swap
  void Swap(GeneralMatrix *other);
 private:
  // We don't explicitly store the type of the matrix.  Rather, we make
  // sure that only one of the matrices is ever nonempty, and the Type()
  // returns that one, or kFullMatrix if all are empty.
  Matrix<BaseFloat> mat_;
  CompressedMatrix cmat_;
  SparseMatrix<BaseFloat> smat_;
};
```

## qr

QR算法

QR algorithm（以下简称QR算法）是一种用递归法求解矩阵特征值和特征方向的算法

## matrix-functions

实现了FFT的计算

```cpp
template<typename Real> 
void ComplexFft (VectorBase<Real> *v, bool forward, Vector<Real> *tmp_work = NULL);
//和FFT一样，但以一种低效的方式实现了傅立叶变换
template<typename Real> 
void ComplexFt (const VectorBase<Real> &in,VectorBase<Real> *out, bool forward);
//RealFft是实输入的傅立叶变换，内部使用ComplexFft。
template<typename Real> void RealFft (VectorBase<Real> *v, bool forward);
//RealFt具有与上述RealFft相同的输入和输出格式，但是它出于测试目的而效率低下。
template<typename Real> void RealFftInefficient (VectorBase<Real> *v, bool forward);
//ComputeDctMatrix计算与DCT对应的矩阵
template<typename Real> void ComputeDctMatrix(Matrix<Real> *M);
//ComplexMul实现复数乘法b * = a。
template<typename Real> 
inline void ComplexMul(const Real &a_re, const Real &a_im,Real *b_re, Real *b_im);
template<typename Real> 
//ComplexMul内联实现c + =（a * b）复杂操作。
inline void ComplexAddProduct(const Real &a_re, const Real &a_im,
                     const Real &b_re, const Real &b_im,Real *c_re, Real *c_im);
//ComplexImExp implements a <-- exp(i x), inline.
template<typename Real> inline void ComplexImExp(Real x, Real *a_re, Real *a_im);
//主成分分析，旨在利用降维的思想，把多指标转化为少数几个综合指标。
template<typename Real>
void ComputePca(const MatrixBase<Real> &X,MatrixBase<Real> *U,MatrixBase<Real> *A,
                bool print_eigs = false,bool exact = true);
//*plus += max(0, a b^T),*minus += max(0, -(a b^T))
template<typename Real>
void AddOuterProductPlusMinus(Real alpha,const VectorBase<Real> &a,
               const VectorBase<Real> &b,MatrixBase<Real> *plus,MatrixBase<Real> *minus);
```

## srfft

此类基于Henrique（Rico）Malvar的代码（来自“重叠变换的信号处理”）（1992年）。 经许可，由Go Vivace Inc.优化，并由Microsoft Corporation转换为C ++
与ComplexFft（在matrix-functios.h中声明）相比，这是执行复数FFT的一种更有效的方法，但是它仅适用于2的幂。
注意：在多线程代码中，每个线程需要具有这些对象之一，因为并行调用Compute无效。

## optimization

优化器，包含：

LinearCgd：线性共轭梯度下降

OptimizeLbfgs：L-BFGS优化算法

