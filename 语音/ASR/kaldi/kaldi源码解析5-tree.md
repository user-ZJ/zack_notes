# kaldi源码解析5-tree

kaldi-tree是kaldi中决策树的实现。

决策树是一种树形结构，其中每个内部节点表示一个属性上的判断，每个分支代表一个判断结果的输出，最后每个叶节点代表一种分类结果。

## event-map

决策树主体的定义

```cpp
typedef int32 EventKeyType;
typedef int32 EventValueType;
typedef int32 EventAnswerType;
typedef std::vector<std::pair<EventKeyType, EventValueType> > EventType;  //决策条件列表
```

EventMap表示从EventType到仅一个整数的EventAnswerType的映射

```cpp
class EventMap {
 public:
  static void Check(const EventType &event);//will crash if not sorted and unique on key.
  //查询event的下一步key所对应的ans
  static bool Lookup(const EventType &event, EventKeyType key, EventValueType *ans);
  // Maps events to the answer type. input must be sorted.
  virtual bool Map(const EventType &event, EventAnswerType *ans) const = 0;
  // MultiMap将部分指定的事件集映射到它可能映射到的答案集
  virtual void MultiMap(const EventType &event, std::vector<EventAnswerType> *ans) const = 0;

  // 返回作为该EventMap的直接子级的EventMap（如果存在）。 对于确定事件图的结构很有用。
  virtual void GetChildren(std::vector<EventMap*> *out) const = 0;
  // event map的深拷贝
  virtual EventMap *Copy(const std::vector<EventMap*> &new_leaves) const = 0;
  EventMap *Copy() const { std::vector<EventMap*> new_leaves; return Copy(new_leaves); }

  // 用于在不同的整数表示形式之间映射音素集合。
  virtual EventMap *MapValues(
      const unordered_set<EventKeyType> &keys_to_map,
      const unordered_map<EventValueType,EventValueType> &value_map) const = 0;

  // 类似于Copy（），不同之处在于它删除树的仅返回-1的部分
  virtual EventMap *Prune() const = 0;
  virtual EventAnswerType MaxResult() const {  // child classes may override this for efficiency; here is basic version.
    // returns -1 if nothing found.
    std::vector<EventAnswerType> tmp; EventType empty_event;
    MultiMap(empty_event, &tmp);
    if (tmp.empty()) {
      KALDI_WARN << "EventMap::MaxResult(), empty result";
      return std::numeric_limits<EventAnswerType>::min();
    }
    else { return * std::max_element(tmp.begin(), tmp.end()); }
  }
  virtual void Write(std::ostream &os, bool binary) = 0;
  virtual ~EventMap() {}
  /// a Write function that takes care of NULL pointers.
  static void Write(std::ostream &os, bool binary, EventMap *emap);
  /// a Read function that reads an arbitrary EventMap; also
  /// works for NULL pointers.
  static EventMap *Read(std::istream &is, bool binary);
};
```

ConstantEventMap：静态决策树

TableEventMap：决策树列表

SplitEventMap：树拆分，如将一个节点拆分为多个节点

## build-tree

构建决策树，返回EventMap

```cpp
//BuildTree是构建一组决策树的常规方法。 集合“ phone_sets”决定了我们如何建立决策树的根。 每个音素集合phone_sets [i]具有共享的决策树根，并且如果相应的变量share_roots [i]为true，则将为音素中不同的HMM位置共享根。 “ phone_sets”中的所有音素都应处于统计信息中（请使用FixUnseenPhones来确保这一点）。 如果对于任何i，do_split [i]为false，我们将不对该集合中的音素进行任何树拆分。
EventMap *BuildTree(Questions &qopts,
             const std::vector<std::vector<int32> > &phone_sets,
             const std::vector<int32> &phone2num_pdf_classes,
             const std::vector<bool> &share_roots,
             const std::vector<bool> &do_split,
             const BuildTreeStatsType &stats,
             BaseFloat thresh,
             int32 max_leaves,
             BaseFloat cluster_thresh,//typically==thresh.If negative,use smallest split.
             int32 P, 
             bool round_num_leaves = true);
//BuildTreeTwoLevel生成一个两级树，例如，在构建具有多个代码本的捆绑混合系统时很有用。 首先通过拆分为“ max_leaves_first”来构建一棵小树。 然后，它在“ max_leaves_first”的叶子处分裂（将其想象为在第一棵树的叶子处创建多个小树），直到叶子总数达到“ max_leaves_second”。 然后，它输出第二棵树，以及从第二棵树的叶ID到第一棵树的叶ID的映射。 请注意，该接口类似于BuildTree，实际上它在内部调用BuildTree。
EventMap *BuildTreeTwoLevel(Questions &qopts,
                            const std::vector<std::vector<int32> > &phone_sets,
                            const std::vector<int32> &phone2num_pdf_classes,
                            const std::vector<bool> &share_roots,
                            const std::vector<bool> &do_split,
                            const BuildTreeStatsType &stats,
                            int32 max_leaves_first,
                            int32 max_leaves_second,
                            bool cluster_leaves,
                            int32 P,
                            std::vector<int32> *leaf_map);
//GenRandStats生成BuildTree使用的随机统计信息表，GenRandStats仅对测试有用，但可用于记录BuildTreeDefault使用的统计信息格式。
void GenRandStats(int32 dim, int32 num_stats, int32 N, int32 P,
                  const std::vector<int32> &phone_ids,
                  const std::vector<int32> &hmm_lengths,
                  const std::vector<bool> &is_ctx_dep,
                  bool ensure_all_phones_covered,
                  BuildTreeStatsType *stats_out);
//读取OpenFst符号表，丢弃符号并输出整数
void ReadSymbolTableAsIntegers(std::string filename,
                               bool include_eps,
                               std::vector<int32> *syms);
//输出合理的音素集，以便在造树算法中提出问题。 这些是通过音素的树状群集获得的； 对于树中的每个节点，从该节点可访问的所有叶子都构成了一组音素。
void AutomaticallyObtainQuestions(BuildTreeStatsType &stats,
                                  const std::vector<std::vector<int32> > &phone_sets_in,
                                  const std::vector<int32> &all_pdf_classes_in,
                                  int32 P,
                                  std::vector<std::vector<int32> > *questions_out);
//此功能使用k-means算法将音素聚集到音素组中。 例如，在为适应目的而构建简单模型时很有用。
void KMeansClusterPhones(BuildTreeStatsType &stats,
                         const std::vector<std::vector<int32> > &phone_sets_in,
                         const std::vector<int32> &all_pdf_classes_in,
                         int32 P,
                         int32 num_classes,
                         std::vector<std::vector<int32> > *sets_out);
//读取根文件
void ReadRootsFile(std::istream &is,
                   std::vector<std::vector<int32> > *phone_sets,
                   std::vector<bool> *is_shared_root,
                   std::vector<bool> *is_split_root);
```

## clusterable-classes

ScalarClusterable：标量聚类，loss为x^2

GaussClusterable:高斯聚类

VectorClusterable：向量聚类，每个向量都与一个权重相关联，目标函数（要最大化）是从聚类中心到每个向量的距离的平方和的负值，乘以该向量的权重。

## tree-renderer

解析决策树文件并以GraphViz格式输出其描述



