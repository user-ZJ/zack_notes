# kaldi  FST

### WFST

WFST由一组状态（State）和状态间的有向跳转（Transition）构成，其中每个跳转上保存由三种信息（输入标签，输出标签，权重）

WFST还应具备一个起始状态（Initial state）和至少一个终止状态（Final state）；每个终止状态可以由一个终止权重（Final weight）

习惯上用**epsilon**代表空标签

### OpenFst

OpenFst遵循Apache协议的开源软件，提供了WSFT的表示和各种操作的实现，以C++ API形式和二进制可执行文件形式提供。OpenFst定义了一种WFST的描述语言，用该语言可以描述一个WFST。比如example.fst.txt

```
0 1 d data 1.0
0 5 d dew  1.0
1 2 ey <esp> 0.5
1 2 ae <esp> 0.5
2 3 t <esp> 0.3
2 3 dx <esp> 0.7
3 4 ax <esp> 1.0
5 6 uw <esp> 1.0
4 1.0
6 1.0
```

除最后两行外，每行代表一个跳转，由5个元素构成，分别代表该跳转的**源状态、目标状态，输入标签、输出标签和权重**。最后两行表示终止状态，终止权重为1.0

example.fst.txt表示的WFST，如果一个输入的序列是“d ey t ax”,那么输出的序列就是“data esp esp esp”,一般esp省略，即输出序列可认为是“data”,权重为路径上的权重之和（或其他自定义权重计算方法）

Openfst要求输入标签和输出标签都用数字表示，因此还需要定义一个标签文本到数字的映射表，这里保存为word.txt

```
<esp> 0
d 1
data 2
dew 3
ey 4
ae 5
t 6
dx 7
ax 8
uw 9
```

可以使用OpenFst的编译工具将example.fst.txt编译成而进制文件

```shell
fstcompile --isymbols=word.txt --osymbols=word.txt example.fst.txt example.fst
```

isymbols表示输入标签列对应的id，osymbols表示输出标签列对应的id，也可以将两列整合到一个映射表中。

编译成二进制后，就可以使用Openfst的工具对其进行各种操作了。

fstinfo 查看fst信息

fstprint 将fst打印成文本形式

fstdraw 将fst输出成Graphviz软件定义的图格式（dot格式）以便可视化。通过dot命令转为ps格式，然后可以由ps2pdf命令转为pdf文档

```shell
fstdraw HCLG.fst | dot -Tps | ps2pdf - HCLG.pdf

fstdraw --isymbols=words.txt --osymbols=words.txt G.fst | dot -Tjpg > fst.jpg
```

```cpp
template <class W>
struct ArcTpl {
 public:
  using Weight = W;
  using Label = int;
  using StateId = int;

  Label ilabel;
  Label olabel;
  Weight weight;
  StateId nextstate;

  ArcTpl() {}

  ArcTpl(Label ilabel, Label olabel, Weight weight, StateId nextstate)
      : ilabel(ilabel),
        olabel(olabel),
        weight(std::move(weight)),
        nextstate(nextstate) {}

  static const string &Type() {
    static const string *const type =
        new string(Weight::Type() == "tropical" ? "standard" : Weight::Type());
    return *type;
  }
};
```





### 令牌（token）

令牌实际上是历史路径的记录。每一个令牌都可以读出或回溯出全部的历史路径信息。令牌上还存储该路径的累积代价，用于评估该路径的优劣。

如果一个状态有多个跳转，那么就把令牌复制多份，分别传递。这样传递到最后一帧，检查所有令牌的代价，选出一个最优令牌，就是搜索路径搜索结果了。如果选分数排名靠前的若干令牌，就得到了N-best结果 

```cpp
struct StdToken {
  using ForwardLinkT = ForwardLink<StdToken>;
  using Token = StdToken;

  // Each active HCLG (decoding-graph) state on each frame has one token.
  //每帧上的每个活动 HCLG（解码图）状态都有一个令牌

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  //从开始到当前状态的总代价（语言模型+声学模型）
  BaseFloat tot_cost;

  // exta_cost>= 0。在调用 PruneForwardLinks 之后，这等于此链接所属的最佳路径的成本与绝对最佳路径的成本之间的最小差值
  BaseFloat extra_cost;

  // 令牌中存储的历史路径的头部，用来生产lattice
  ForwardLinkT *links;

  //'next' 是该帧的单链标记列表中的下一个。
  Token *next;

  // 这个函数什么都不做，应该优化掉； 
  // 它是必需的，由此我们可以共享常规的 LatticeFasterDecoderTpl 代码和支持快速回溯的 LatticeFasterOnlineDecoder 代码。
  inline void SetBackpointer (Token *backpointer) { }

  // 这个构造函数只是忽略了“backpointer”参数。 
  // 需要该参数，以便我们可以对 LatticeFasterDecoderTpl 和 LatticeFasterOnlineDecoderTpl 使用相同的解码器代码
  //（LatticeFasterOnlineDecoderTpl需要反向指针来支持获得最佳路径的快速方法）。
  inline StdToken(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLinkT *links,
                  Token *next, Token *backpointer):
      tot_cost(tot_cost), extra_cost(extra_cost), links(links), next(next) { }
};
```

### lattice

在kaldi中word lattice(词格)被定义为一个特殊的WFST，该WFST的每个跳转的权重由两个值构成，不是标准WFST的一个值。这两个值分别代表声学分数和语言分数。和HCLG一样，词格的输入标签和输出标签分别是transition-id和word-id。

```
源状态、目标状态，输入标签、输出标签和权重(graph cost，acoustic cost)
```



kaldi中词格满足以下特性：

* 所有解码分数或负代价大于某阈值的输出标签（单词）序列，都可以在词格中找到对应的路径
* 词格中每条路径的分数和输入标签序列都能在HCLG中找到对应的路径
* 对于任意输出标签序列，最多只能在词格中找到一条路径

可以把词格想象成一个简化的状态图，其中只包含解码分数较高的路径，而去除了原图中可能性较小的路径。同时，把解码时计算的声学分数记录到了这些路径中。这样，词格就可以作为解码的结果，即包含了最佳路径，也包含了其他可选路径。

```cpp
typedef fst::CompactLatticeWeightTpl<LatticeWeight, int32> CompactLatticeWeight;  // {{graph cost，acoustic cost},{transion_id的序列}}
typedef fst::ArcTpl<CompactLatticeWeight> CompactLatticeArc;
typedef fst::VectorFst<CompactLatticeArc> CompactLattice;
typedef fst::LatticeWeightTpl<BaseFloat> LatticeWeight;  //{graph cost，acoustic cost}
typedef fst::ArcTpl<LatticeWeight> LatticeArc;  //{ilabel,olabel,LatticeWeight,to_state}
typedef fst::VectorFst<LatticeArc> Lattice;
```

```cpp
0	2	0	0	3.22852,0
0	1	0	0
1	1899	7016	152797	13.3926,28.3982
1	1898	7015	0	9.77051,23.9595
1	1897	7003	0	10.7998,26.5885
1	1896	6999	0	11.3848,28.9709
1	1895	6991	152350	12.3643,25.9589
1	1894	6976	152282	13.2119,25.7399
1	1893	6973	0	9.19434,23.6919
```



### CompactLattice

包含和lattice相同的信息，但形式不同。

它是一个 acceptor（意味着输入输出符号始终相同），输入输出符号代表词，而 transition-ids序列作为了权重的一部分（注意 OpenFst有一个很通用的权重概念；满足半环公理的都可以作为权值）。CompactLattice 中的权重包括一对浮点数和一个整数序列（代表 transition-ids）

对于需要在lattice和compactlattice之间进行转换的程序，可以使用 [ConvertLattice()](https://shiweipku.gitbooks.io/chinese-doc-of-kaldi/content/lattice.html)

```cpp
// 源状态、目标状态，word-id,权重（graph cost，acoustic cost，transion_id）
0	3425	0	3.22852,0,
0	1	0
1	17044	152797	13.3926,28.3982,7016
1	3421	0	9.77051,23.9595,7015
1	3420	0	10.7998,26.5885,7003
1	3419	0	11.3848,28.9709,6999
1	3418	152350	12.3643,25.9589,6991
1	3417	152282	13.2119,25.7399,6976
1	3413	0	9.19434,23.6919,6973
1	3412	151889	10.0615,28.3522,6970
1	3408	0	8.58398,24.008,6968
1	16950	151440	12.4844,28.8381,6964
1	3407	151386	9.91602,25.12,6962
```



### DecodableInterface

Kaldi中抽象出DecodableInterface，定义了kaldi中所有解码器的声学分来源，将解码器算法和声学模型计算模块解耦合，使解码器不依赖声学模型的内部机理。

无论声学分来源GMM还是nnet2、net3，或者类似TensorFlow、pytorch等深度学习框架，甚至事先生成好的声学分矩阵，解码器都可以一视同仁的解码。

```cpp
class DecodableInterface {
 public:
  /// 返回某一帧frame的某一个状态index的对数似然值
  virtual BaseFloat LogLikelihood(int32 frame, int32 index) = 0;

  ///判断是否已经取到了最后一帧
  virtual bool IsLastFrame(int32 frame) const = 0;

  ///返回已经可用的帧数
  virtual int32 NumFramesReady() const {
    KALDI_ERR << "NumFramesReady() not implemented for this decodable type.";
    return -1;
  }

  ///返回状态个数
  virtual int32 NumIndices() const = 0;

  virtual ~DecodableInterface() {}
};
```



输入标签不为esp的跳转称为emitting跳转

输入标签为esp的跳转称为non-emitting跳转



ProcessEmitting  对HCLG上emitting跳转进行令牌传递

ProcessNonemitting  对HCLG上non-emitting跳转的令牌进行传递

PruneToks  对令牌进行剪枝



```cpp
using Elem = typename HashList<StateId, Token*>::Elem;                                                      
// Equivalent to:
//  struct Elem {
//    StateId key;
//    Token *val;
//    Elem *tail;
//  };
//每帧令牌列表的头部（列表按拓扑顺序排列），以及说明我们是否曾经使用 PruneForwardLinks 对其进行修剪的内容。
struct TokenList {
  Token *toks;
  bool must_prune_forward_links;
  bool must_prune_tokens;
  TokenList(): toks(NULL), must_prune_forward_links(true),must_prune_tokens(true) { }
};

//每帧所拥有的token链表表头
std::vector<TokenList> active_toks_;
//所有有token的state的hashlist
//FindOrAddToken会添加令牌到toks_
HashList<StateId, Token*> toks_;
//存在ilabel==0的边的state
std::vector<const Elem* > queue_; 



//令牌到下一帧令牌的链接，或有时在当前帧上（用于输入 epsilon 链接）
template <typename Token>
struct ForwardLink {
  using Label = fst::StdArc::Label;

  Token *next_tok;  // the next token [or NULL if represents final-state]
  Label ilabel;  // ilabel on arc
  Label olabel;  // olabel on arc
  BaseFloat graph_cost;  // graph cost of traversing arc (contains LM, etc.)
  BaseFloat acoustic_cost;  // acoustic cost (pre-scaled) of traversing arc
  ForwardLink *next;  // next in singly-linked list of forward arcs (arcs
                      // in the state-level lattice) from a token.
  inline ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                     BaseFloat graph_cost, BaseFloat acoustic_cost,
                     ForwardLink *next):
      next_tok(next_tok), ilabel(ilabel), olabel(olabel),
      graph_cost(graph_cost), acoustic_cost(acoustic_cost),
      next(next) { }
};



using ForwardLinkT = decoder::ForwardLink<Token>;

//对于每一帧，包含一个偏移量，该偏移量被添加到该帧的声学对数似然中，以便将所有内容保持在良好的动态范围内，即接近于零，以减少舍入误差。
std::vector<BaseFloat> cost_offsets_;  
```

```cpp
//在toks_的哈希链表中定位一个令牌，或者在没有前向链接(forward links)时为当前帧插入一个新的空令牌
//frame_plus_one 参数是声学帧索引加一，用于索引到 active_toks_ 数组
//注意：如有必要，它会插入到哈希 toks_ 中，也插入到此帧上活动的标记的单向链接列表中（其头部位于 active_toks_[frame]）
//返回令牌指针。 如果令牌是新创建的或cost已更改，则将“changed”（如果非 NULL）设置为 true。 如果 Token == StdToken，则 'backpointer' 参数没有意义（并且有望被优化掉）。
inline Elem *FindOrAddToken(StateId state, int32 frame_plus_one,
                              BaseFloat tot_cost, Token *backpointer,
                              bool *changed);
// 获取权重截断值。 还计算活动令牌。
BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem);
//保证toks_的hashlist长度为num_toks的2倍以上
void PossiblyResizeHash(size_t num_toks);
// 删除token中ForwardLinkT *links链表中的所有元素
inline static void DeleteForwardLinks(Token *tok);
```



## lattice-faster-decoder.cc

```cpp
template <typename FST, typename Token>
BaseFloat LatticeFasterDecoderTpl<FST, Token>::ProcessEmitting(
    DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size() > 0);
  int32 frame = active_toks_.size() - 1; // 帧索引，从0开始计数，用于从decodable获取likelihoods
  active_toks_.resize(active_toks_.size() + 1);

  Elem *final_toks = toks_.Clear(); //从hashlist中获取所有令牌的头指针，会导致toks_的hash表为空，final_toks中所有元素由ProcessEmitting函数来管理
  Elem *best_elem = NULL;
  BaseFloat adaptive_beam;
  size_t tok_cnt; //活动令牌数
  //计算权重截断值和活动令牌数,并将最优令牌节点赋值给best_elem
  BaseFloat cur_cutoff = GetCutoff(final_toks, &tok_cnt, &adaptive_beam, &best_elem);
  KALDI_VLOG(6) << "Adaptive beam on frame " << NumFramesDecoded() << " is "
                << adaptive_beam;
  //保证toks_的hashlist长度为tok_cnt的2倍以上
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.

  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens

  BaseFloat cost_offset = 0.0; // Used to keep probabilities in a good dynamic range.

  // 处理当前最优的节点，以便在下一个截止点上获取最优路径
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    cost_offset = - tok->tot_cost;
    for (fst::ArcIterator<FST> aiter(*fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // 仅处理下一跳不为esp的边
        BaseFloat new_weight = arc.weight.Value() + cost_offset -
            decodable->LogLikelihood(frame, arc.ilabel) + tok->tot_cost;
        if (new_weight + adaptive_beam < next_cutoff)
          next_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // 存储声学概率分的offset
  cost_offsets_.resize(frame + 1, 0.0);
  cost_offsets_[frame] = cost_offset;

  // final_toks为所有token的头节点
  for (Elem *e = final_toks, *e_tail; e != NULL; e = e_tail) {
    // loop this way because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    if (tok->tot_cost <= cur_cutoff) {
      for (fst::ArcIterator<FST> aiter(*fst_, state);
           !aiter.Done();
           aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) {  // 仅处理下一跳不为esp的边
          BaseFloat ac_cost = cost_offset -
              decodable->LogLikelihood(frame, arc.ilabel),  //声学概率分
              graph_cost = arc.weight.Value(),
              cur_cost = tok->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost;
          if (tot_cost >= next_cutoff) continue;
          else if (tot_cost + adaptive_beam < next_cutoff)
            next_cutoff = tot_cost + adaptive_beam; // prune by best current token
          //给下一个节点创建一个令牌
          //使用activate_toks_记录下一帧的活跃令牌
          Elem *e_next = FindOrAddToken(arc.nextstate,
                                        frame + 1, tot_cost, tok, NULL);

          // 将下一个节点的令牌链接到当前令牌的links
          tok->links = new ForwardLinkT(e_next->val, arc.ilabel, arc.olabel,
                                        graph_cost, ac_cost, tok->links);
        }
      } // for all arcs
    }
    //移动到写一个节点，删除上一个节点
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
  return next_cutoff;
}
```



```cpp
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame = static_cast<int32>(active_toks_.size()) - 2;

  // 递归处理一帧的 nonemitting 边 

  KALDI_ASSERT(queue_.empty());

  if (toks_.GetList() == NULL) {
    if (!warned_) {
      KALDI_WARN << "Error, no surviving tokens: frame is " << frame;
      warned_ = true;
    }
  }
  // 遍历所有保存有令牌的状态节点，将边存在ilabel==0的节点加入到queue_
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
    StateId state = e->key;
    if (fst_->NumInputEpsilons(state) != 0)
      queue_.push_back(e);
  }
  //遍历所有存在ilabel==0的节点
  while (!queue_.empty()) {
    const Elem *e = queue_.back();
    queue_.pop_back();

    StateId state = e->key;
    Token *tok = e->val;  // would segfault if e is a NULL pointer but this can't happen.
    BaseFloat cur_cost = tok->tot_cost;
    if (cur_cost >= cutoff) // Don't bother processing successors.
      continue;
    //如果令牌的前向链接（links）不为空，则删除令牌的所有前向链接（links） 
    //后续会重建令牌的前向链接
    DeleteForwardLinks(tok); // necessary when re-visiting
    tok->links = NULL;
    //遍历节点的所有边
    for (fst::ArcIterator<FST> aiter(*fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // 如果ilabel==0,即输入为esp
        BaseFloat graph_cost = arc.weight.Value(),
            tot_cost = cur_cost + graph_cost;
        if (tot_cost < cutoff) {
          bool changed;
		  //给下一个节点创建一个令牌
          //使用activate_toks_记录下一帧的活跃令牌
          Elem *e_new = FindOrAddToken(arc.nextstate, frame + 1, tot_cost,
                                          tok, &changed);
          // 将下一个节点的令牌链接到当前令牌的links
          tok->links = new ForwardLinkT(e_new->val, 0, arc.olabel,
                                        graph_cost, 0, tok->links);

          // 递归检查
          // 如果下一个状态的令牌是新创建的，或者修改了原有令牌的cost，且下一个状态中有ilabel==0的边，则将queue_
          if (changed && fst_->NumInputEpsilons(arc.nextstate) != 0)
            queue_.push_back(e_new);
        }
      }
    } // for all arcs
  } // while queue not empty
}
```

```cpp
template <typename FST, typename Token>
inline typename LatticeFasterDecoderTpl<FST, Token>::Elem*
LatticeFasterDecoderTpl<FST, Token>::FindOrAddToken(
      StateId state, int32 frame_plus_one, BaseFloat tot_cost,
      Token *backpointer, bool *changed) {
  // 返回token指针.  如果token是新建的，或改变了token的cost，设置changed为true
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  Elem *e_found = toks_.Insert(state, NULL);
  if (e_found->val == NULL) {  // 当前state不存在token
    const BaseFloat extra_cost = 0.0;
    // 创建新token，并链接该帧的active_toks_链表
    Token *new_tok = new Token (tot_cost, extra_cost, NULL, toks, backpointer);
    // NULL: no forward links yet
    toks = new_tok;
    num_toks_++;
    e_found->val = new_tok;  //token和当前节点绑定
    if (changed) *changed = true;
    return e_found;
  } else {  //当前state存在token
    Token *tok = e_found->val;  // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) {  // 原有token中的cost大于新路劲的cost，更新token
      tok->tot_cost = tot_cost;
      // SetBackpointer() just does tok->backpointer = backpointer in
      // the case where Token == BackpointerToken, else nothing.
      tok->SetBackpointer(backpointer);
      // we don't allocate a new token, the old stays linked in active_toks_
      // we only replace the tot_cost
      // in the current frame, there are no forward links (and no extra_cost)
      // only in ProcessNonemitting we have to delete forward links
      // in case we visit a state for the second time
      // those forward links, that lead to this replaced token before:
      // they remain and will hopefully be pruned later (PruneForwardLinks...)
      if (changed) *changed = true;
    } else {
      if (changed) *changed = false;
    }
    return e_found;
  }
}
```

```cpp
//初始化解码器
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::InitDecoding() {
  // clean up from last time:
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();   //cost归一化
  ClearActiveTokens();  //清除所有token
  warned_ = false;  
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  StateId start_state = fst_->Start();  //重置fst状态为0
  KALDI_ASSERT(start_state != fst::kNoStateId);
  active_toks_.resize(1);   // 添加状态为0的节点的token，并处理所有非发射状态
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, NULL);
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_state, start_tok);
  num_toks_++;
  ProcessNonemitting(config_.beam);
}
```

```cpp
// 实际解码函数
// decodable 用于获取每帧的声学模型权重
// max_num_frames要解码的帧数，如果为-1则解码全部帧
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::AdvanceDecoding(DecodableInterface *decodable,
                                                int32 max_num_frames) {
  if (std::is_same<FST, fst::Fst<fst::StdArc> >::value) {
    // if the type 'FST' is the FST base-class, then see if the FST type of fst_
    // is actually VectorFst or ConstFst.  If so, call the AdvanceDecoding()
    // function after casting *this to the more specific type.
    if (fst_->Type() == "const") {
      LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, Token>* >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    } else if (fst_->Type() == "vector") {
      LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, Token>* >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    }
  }


  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");
  // 声学模型完成解码的帧总数
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded；NumFramesDecoded()获取完成wfst解码的帧数
  KALDI_ASSERT(num_frames_ready >= NumFramesDecoded());
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     NumFramesDecoded() + max_num_frames);
  // 循环解码
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);  //剪枝
    }
    BaseFloat cost_cutoff = ProcessEmitting(decodable);  // 处理发射状态
    ProcessNonemitting(cost_cutoff);  // 处理非发射状态
  }
}
```

