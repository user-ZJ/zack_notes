### run_tdnn.sh

```shell
. ./cmd.sh

# 提取mfcc特征和倒谱信息
mfccdir=mfcc
for x in train dev test; do
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 10 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  utils/fix_data_dir.sh data/$x || exit 1;
done

# 训练单音素HMM模型
# data/train为训练数据wav.scp,text.utt2spk;
# data/lang为事先准备好的和语言模型相关的文件，由prepare_lang.sh生成，\
# L_disambig.fst,L.fst,oov.int,oov.txt,phones.txt,topo,words.txt等文件
# exp/mono 输出文件路径
steps/train_mono.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/mono || exit 1;

# 单音素模型解码
# 调用utils/mkgraph.sh；来对刚刚生成的声学文件进行构图
utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/mono/graph data/dev exp/mono/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/mono/graph data/test exp/mono/decode_test

# 单音素模型.
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/mono exp/mono_ali || exit 1;

# 训练与上下文相关的三音 HMM 模型 tri1
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# 解码 tri1
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri1/graph data/dev exp/tri1/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri1/graph data/test exp/tri1/decode_test

# 对齐 tri1
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# 训练 tri2 [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

# 解码 tri2
utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri2/graph data/dev exp/tri2/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri2/graph data/test exp/tri2/decode_test

# train and decode tri2b [LDA+MLLT]
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# 进行线性判别分析(LDA)和最大似然线性转换(MLLT),训练 tri3a, which is LDA+MLLT,
steps/train_lda_mllt.sh --cmd "$train_cmd" \
 2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri3a/graph data/dev exp/tri3a/decode_dev
steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri3a/graph data/test exp/tri3a/decode_test

# 从现在开始，我们将开始构建更严格的系统（使用SAT），并使用fMLLR进行调整。

steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;
# 训练发音人自适应，基于特征空间最大似然线性回归
steps/train_sat.sh --cmd "$train_cmd" \
  2500 20000 data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri4a/graph data/dev exp/tri4a/decode_dev
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri4a/graph data/test exp/tri4a/decode_test

steps/align_fmllr.sh  --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/tri4a exp/tri4a_ali

# 构建一个大的 SAT 系统.

steps/train_sat.sh --cmd "$train_cmd" \
  3500 100000 data/train data/lang exp/tri4a_ali exp/tri5a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
   exp/tri5a/graph data/dev exp/tri5a/decode_dev || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
   exp/tri5a/graph data/test exp/tri5a/decode_test || exit 1;
#将特征向量进行gmm-est-fmllr并gmm-align-compiled对齐操作
steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;

# nnet3
local/nnet3/run_tdnn.sh

# chain
local/chain/run_tdnn.sh

# getting results (see RESULTS file)
for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
for x in exp/*/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

exit 0;
```

### train_mono.sh

```shell
# train_mono.sh
# Begin configuration section.
nj=4  #并行job数，在训练的job并行训练过程中，训练数据的各个子集合是分散到不同的处理器去进行训练，然后每轮迭代后会进行合并。
cmd=run.pl  #job调度分发脚本，默认是run.pl
# 控制缩放选项
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=40    # 训练迭代次数
max_iter_inc=30 # 每个轮次最大增加的高斯数
initial_beam=6 # 第一次迭代中使用的beam (设置一个较小的数，加快初始化)
regular_beam=10 # 第一次迭代后使用的beam
retry_beam=40
totgauss=1000 # 总高斯数
careful=false
boost_silence=1.0 # boost_silence系数，用来增加静音音素在对齐过程中的似然
# 在第N次迭代进行一次对齐操作
realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38";
config= # name of config file.
stage=-4
power=0.25 # 从出现次数确定高斯数的指数
norm_vars=false # 不推荐使用，建议使用--cmvn-opts“ --norm-vars = false”
cmvn_opts=  # 可用于向cmvn添加其他选项.
delta_opts= # 可用于为添加增量添加额外的选项
# End configuration section.
echo "$0 $@"  # Print the command line for logging

# 参数处理
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: steps/train_mono.sh [options] <data-dir> <lang-dir> <exp-dir>"
  echo " e.g.: steps/train_mono.sh data/train.1k data/lang exp/mono"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data=$1 # 第一个参数是训练数据所在的目录 
lang=$2 # 第二个参数是语言模型所在的目录 
dir=$3  # 日志和最终目标文件输出目录

oov_sym=`cat $lang/oov.int` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
# -d判断是不是目录； filename1-ot filename2 如果 filename1比 filename2旧，则为真
# 把$data文件夹下的各个文件都进行split切分，方便多线程处理
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $lang/phones.txt $dir || exit 1;

$norm_vars && cmvn_opts="--norm-vars=true $cmvn_opts"
echo $cmvn_opts  > $dir/cmvn_opts # keep track of options to CMVN.
[ ! -z $delta_opts ] && echo $delta_opts > $dir/delta_opts # keep track of options to delta

# feats变量定义，这个变量作为后续其他命令的参数，这个主要处理特征数据的
# apply-cmvn –utt2spk=ark:sdata/JOB/utt2spk语料和录音人员关联文件
# scp:sdata/JOB/cmvn.scp 说话人相关的均值和方差 scp:$sdata/JOB/feats.scp 训练用特征文件 
# 对feats.scp做归一化处理,输出是 ark:-|，利用管道技术把输出传递给下一个函数作为输入
# add-deltas 输入是ark:-，表示把上一个函数的输出作为输入（实际上就是整个训练特征数据）
# 训练数据增加差分量，比如13维度mfcc处理后变成39维度，$feats就变成了 ark,s,cs:特征数据
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
example_feats="`echo $feats | sed s/JOB/1/g`";

echo "$0: Initializing monophone system."

[ ! -f $lang/phones/sets.int ] && exit 1;
shared_phones_opt="--shared-phones=$lang/phones/sets.int"

if [ $stage -le -3 ]; then
  # Note: JOB=1 just uses the 1st part of the features-- we only need a subset anyway.
  # 特征维度获取
  if ! feat_dim=`feat-to-dim "$example_feats" - 2>/dev/null` || [ -z $feat_dim ]; then
    feat-to-dim "$example_feats" -
    echo "error getting feature dimension"
    exit 1;
  fi
  #首先是初始化GMM，使用的脚本是/kaldi-trunk/src/gmmbin/gmm-init-mono，输出是0.mdl和tree文件
  # "--train-feats=$feats subset-feats --n=10 ark:- ark:-|"  这个的意思是特征数据中取10个特征用于构造原始模型
  # 可以利用 gmm-info $dir/0.mdl 查看这个模型概要信息 
  $cmd JOB=1 $dir/log/init.log \
    gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
    $dir/0.mdl $dir/tree || exit 1;
fi
# 默认高斯数获取，从模型0.mdl里读取的
numgauss=`gmm-info --print-args=false $dir/0.mdl | grep gaussians | awk '{print $NF}'`
# 根据参数总高斯计算每个循环增减的具体高斯数
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss

if [ $stage -le -2 ]; then
  echo "$0: Compiling training graphs"
  # 使用的脚本是/kaldi-trunk/source/bin/compile-training-graphs，输入是tree,0.mdl和L.fst
  # 输出是fits.JOB.gz，JOB是具体分割是数据，比如分割16分（16个并行），其是在训练过程中构建graph的过程
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/0.mdl  $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: Aligning data equally (pass 0)"
  # 对齐操作align-equal-compiled对每一个训练特征做解码
  # gmm-acc-stats-ali：根据对其信息，计算每个高斯分布的均值和方差，信息最后保存在 0.JOB.acc 里
  $cmd JOB=1:$nj $dir/log/align.0.JOB.log \
    align-equal-compiled "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,t:-  \| \
    gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
    $dir/0.JOB.acc || exit 1;
fi

# In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
# we fail to est "rare" phones and later on, they never align properly.

if [ $stage -le 0 ]; then
  # gmm-est：模型重建，根据原始模型0.mdl 所有0.*.acc文件合并输出 1.mdl
  # gmm-sum-accs:完成多个文件的统计合并
  gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
  rm $dir/0.*.acc
fi

beam=$initial_beam # will change to regular_beam below after 1st pass
# note: using slightly wider beams for WSJ vs. RM.
x=1
while [ $x -lt $num_iters ]; do
  echo "$0: Pass $x"
  if [ $stage -le $x ]; then
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "$0: Aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      # gmm-align-compiled：和上面讲的“align-equal-compiled”类似，做特征数据的对齐，对其信息输出到ali.JOB.gz里。
      # 对其不是每次都做，realign_iters=”1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38”; 在 realign_iters 时才做对齐
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
        "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" \
        || exit 1;
    fi
    # gmm-acc-stats-ali：根据对其信息，计算每个高斯分布的均值和方差，输出$x.JOB.acc文件
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
      $dir/$x.JOB.acc || exit 1;
	#调用新生成的参数(高斯数)重新估计GMM gmm-est：根据上一次的模型，构建新模型 [[x+1].mdl
	#对分散在不同处理器上的结果进行合并，生成.mdl结果，调用脚本gmm-acc-sum
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
  fi
  if [ $x -le $max_iter_inc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  beam=$regular_beam
  x=$[$x+1] #循环变量+1
done

( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir
utils/summarize_warnings.pl $dir/log

steps/info/gmm_dir_info.pl $dir

echo "$0: Done training monophone system in $dir"
# 最后生成的.mdl即为声学模型文件
exit 0
# example of showing the alignments:
# show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/ali.0.gz|" | head -4
```

### utils/mkgraph.sh

```shell
# utils/mkgraph.sh
# 该脚本将创建一个完全扩展的解码图（HCLG），该图表示我们模型中的所有语言模型，发音词典（词典），上下文相关性和HMM结构。 输出是一个有限状态换能器，在输出上有单词ID，在输入上有pdf ID（这些是解析为高斯混合模型的索引）。
set -o pipefail

tscale=1.0
loopscale=0.1

remove_oov=false

for x in `seq 4`; do
  [ "$1" == "--mono" -o "$1" == "--left-biphone" -o "$1" == "--quinphone" ] && shift && \
    echo "WARNING: the --mono, --left-biphone and --quinphone options are now deprecated and ignored."
  [ "$1" == "--remove-oov" ] && remove_oov=true && shift;
  [ "$1" == "--transition-scale" ] && tscale=$2 && shift 2;
  [ "$1" == "--self-loop-scale" ] && loopscale=$2 && shift 2;
done

if [ $# != 3 ]; then
   echo "Usage: utils/mkgraph.sh [options] <lang-dir> <model-dir> <graphdir>"
   echo "e.g.: utils/mkgraph.sh data/lang_test exp/tri1/ exp/tri1/graph"
   echo " Options:"
   echo " --remove-oov       #  If true, any paths containing the OOV symbol (obtained from oov.int"
   echo "                    #  in the lang directory) are removed from the G.fst during compilation."
   echo " --transition-scale #  Scaling factor on transition probabilities."
   echo " --self-loop-scale  #  Please see: http://kaldi-asr.org/doc/hmm.html#hmm_scale."
   echo "Note: the --mono, --left-biphone and --quinphone options are now deprecated"
   echo "and will be ignored."
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

lang=$1  # 语言模型路径
tree=$2/tree  # 模型
model=$2/final.mdl # 模型
dir=$3  # 解码图保存路径

mkdir -p $dir

# If $lang/tmp/LG.fst does not exist or is older than its sources, make it...
# (note: the [[ ]] brackets make the || type operators work (inside [ ], we
# would have to use -o instead),  -f means file exists, and -ot means older than).
# 判断生成图的文件是否存在
required="$lang/L.fst $lang/G.fst $lang/phones.txt $lang/words.txt $lang/phones/silence.csl $lang/phones/disambig.int $model $tree"
for f in $required; do
  [ ! -f $f ] && echo "mkgraph.sh: expected $f to exist" && exit 1;
done

if [ -f $dir/HCLG.fst ]; then
  # detect when the result already exists, and avoid overwriting it.
  must_rebuild=false
  for f in $required; do
    [ $f -nt $dir/HCLG.fst ] && must_rebuild=true
  done
  if ! $must_rebuild; then
    echo "$0: $dir/HCLG.fst is up to date."
    exit 0
  fi
fi


N=$(tree-info $tree | grep "context-width" | cut -d' ' -f2) || { echo "Error when getting context-width"; exit 1; }
P=$(tree-info $tree | grep "central-position" | cut -d' ' -f2) || { echo "Error when getting central-position"; exit 1; }

[[ -f $2/frame_subsampling_factor && "$loopscale" == "0.1" ]] && \
  echo "$0: WARNING: chain models need '--self-loop-scale 1.0'";

if [ -f $lang/phones/nonterm_phones_offset.int ]; then
  if [[ $N != 2  || $P != 1 ]]; then
    echo "$0: when doing grammar decoding, you can only build graphs for left-biphone trees."
    exit 1
  fi
  nonterm_phones_offset=$(cat $lang/phones/nonterm_phones_offset.int)
  nonterm_opt="--nonterm-phones-offset=$nonterm_phones_offset"
  prepare_grammar_command="make-grammar-fst --nonterm-phones-offset=$nonterm_phones_offset - -"
else
  prepare_grammar_command="cat"
  nonterm_opt=
fi

mkdir -p $lang/tmp
trap "rm -f $lang/tmp/LG.fst.$$" EXIT HUP INT PIPE TERM
# Note: [[ ]] is like [ ] but enables certain extra constructs, e.g. || in
# place of -o
# -s如果文件存在且非空 
if [[ ! -s $lang/tmp/LG.fst || $lang/tmp/LG.fst -ot $lang/G.fst || \
      $lang/tmp/LG.fst -ot $lang/L_disambig.fst ]]; then
  fsttablecompose $lang/L_disambig.fst $lang/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstpushspecial > $lang/tmp/LG.fst.$$ || exit 1;
  mv $lang/tmp/LG.fst.$$ $lang/tmp/LG.fst
  fstisstochastic $lang/tmp/LG.fst || echo "[info]: LG not stochastic."
fi

clg=$lang/tmp/CLG_${N}_${P}.fst
clg_tmp=$clg.$$
ilabels=$lang/tmp/ilabels_${N}_${P}
ilabels_tmp=$ilabels.$$
trap "rm -f $clg_tmp $ilabels_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $clg || $clg -ot $lang/tmp/LG.fst \
    || ! -s $ilabels || $ilabels -ot $lang/tmp/LG.fst ]]; then
  fstcomposecontext $nonterm_opt --context-size=$N --central-position=$P \
   --read-disambig-syms=$lang/phones/disambig.int \
   --write-disambig-syms=$lang/tmp/disambig_ilabels_${N}_${P}.int \
    $ilabels_tmp $lang/tmp/LG.fst |\
    fstarcsort --sort_type=ilabel > $clg_tmp
  mv $clg_tmp $clg
  mv $ilabels_tmp $ilabels
  fstisstochastic $clg || echo "[info]: CLG not stochastic."
fi

trap "rm -f $dir/Ha.fst.$$" EXIT HUP INT PIPE TERM
if [[ ! -s $dir/Ha.fst || $dir/Ha.fst -ot $model  \
    || $dir/Ha.fst -ot $lang/tmp/ilabels_${N}_${P} ]]; then
  make-h-transducer $nonterm_opt --disambig-syms-out=$dir/disambig_tid.int \
    --transition-scale=$tscale $lang/tmp/ilabels_${N}_${P} $tree $model \
     > $dir/Ha.fst.$$  || exit 1;
  mv $dir/Ha.fst.$$ $dir/Ha.fst
fi

trap "rm -f $dir/HCLGa.fst.$$" EXIT HUP INT PIPE TERM
if [[ ! -s $dir/HCLGa.fst || $dir/HCLGa.fst -ot $dir/Ha.fst || \
      $dir/HCLGa.fst -ot $clg ]]; then
  if $remove_oov; then
    [ ! -f $lang/oov.int ] && \
      echo "$0: --remove-oov option: no file $lang/oov.int" && exit 1;
    clg="fstrmsymbols --remove-arcs=true --apply-to-output=true $lang/oov.int $clg|"
  fi
  fsttablecompose $dir/Ha.fst "$clg" | fstdeterminizestar --use-log=true \
    | fstrmsymbols $dir/disambig_tid.int | fstrmepslocal | \
     fstminimizeencoded > $dir/HCLGa.fst.$$ || exit 1;
  mv $dir/HCLGa.fst.$$ $dir/HCLGa.fst
  fstisstochastic $dir/HCLGa.fst || echo "HCLGa is not stochastic"
fi

trap "rm -f $dir/HCLG.fst.$$" EXIT HUP INT PIPE TERM
if [[ ! -s $dir/HCLG.fst || $dir/HCLG.fst -ot $dir/HCLGa.fst ]]; then
  add-self-loops --self-loop-scale=$loopscale --reorder=true $model $dir/HCLGa.fst | \
    $prepare_grammar_command | \
    fstconvert --fst_type=const > $dir/HCLG.fst.$$ || exit 1;
  mv $dir/HCLG.fst.$$ $dir/HCLG.fst
  if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
    # No point doing this test if transition-scale not 1, as it is bound to fail.
    fstisstochastic $dir/HCLG.fst || echo "[info]: final HCLG is not stochastic."
  fi
fi

# note: the empty FST has 66 bytes.  this check is for whether the final FST
# is the empty file or is the empty FST.
if ! [ $(head -c 67 $dir/HCLG.fst | wc -c) -eq 67 ]; then
  echo "$0: it looks like the result in $dir/HCLG.fst is empty"
  exit 1
fi

# save space.
rm $dir/HCLGa.fst $dir/Ha.fst 2>/dev/null || true

# keep a copy of the lexicon and a list of silence phones with HCLG...
# this means we can decode without reference to the $lang directory.


cp $lang/words.txt $dir/ || exit 1;
mkdir -p $dir/phones
cp $lang/phones/word_boundary.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
cp $lang/phones/align_lexicon.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
cp $lang/phones/optional_silence.* $dir/phones/ 2>/dev/null # might be needed for analyzing alignments.
    # but ignore the error if it's not there.


cp $lang/phones/disambig.{txt,int} $dir/phones/ 2> /dev/null
cp $lang/phones/silence.csl $dir/phones/ || exit 1;
cp $lang/phones.txt $dir/ 2> /dev/null # ignore the error if it's not there.

am-info --print-args=false $model | grep pdfs | awk '{print $NF}' > $dir/num_pdfs
```

### steps/decode.sh

```shell
# steps/decode.sh
# Begin configuration section.
transform_dir=   # this option won't normally be used, but it can be used if you want to
                 # supply existing fMLLR transforms when decoding.
iter=
model= # You can specify the model to use (e.g. if you want to use the .alimdl)
stage=0
nj=4
cmd=run.pl
max_active=7000
beam=13.0
lattice_beam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
parallel_opts=  # ignored now.
scoring_opts=
# note: there are no more min-lmwt and max-lmwt options, instead use
# e.g. --scoring-opts "--min-lmwt 1 --max-lmwt 20"
skip_scoring=false
decode_extra_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/decode.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.alimdl)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   echo "  --num-threads <n>                                # number of threads to use, default 1."
   echo "  --parallel-opts <opts>                           # ignored now, present for historical reasons."
   exit 1;
fi


graphdir=$1 # 解码图
data=$2  # 测试数据
dir=$3  # 输出测试结果文件夹
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj; #切分文件夹

mkdir -p $dir/log #创建结果文件夹/log文件夹
# $sdata文件夹是否存在 且 里面的feats.scp是否比该文件夹旧;文件夹切分
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl; #$iter也没有指定迭代次数，则模型文件赋值
  else model=$srcdir/$iter.mdl; fi #如果$iter有指定，则模型文件重新赋值;
fi

if [ $(basename $model) != final.alimdl ] ; then #如模型文件不为 final.alidl
  # Do not use the $srcpath -- look at the path where the model is
  # #如果/final.alimdl不存在，且transform_dir没有指定
  if [ -f $(dirname $model)/final.alimdl ] && [ -z "$transform_dir" ]; then
    echo -e '\n\n'
    echo $0 'WARNING: Running speaker independent system decoding using a SAT model!'
    echo $0 'WARNING: This is OK if you know what you are doing...'
    echo -e '\n\n'
  fi
fi

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fst; do
  #以上文件如果不存在，则 打印提示，退出
  [ ! -f $f ] && echo "$0: Error: no such file $f" && exit 1; 
done

# 根据final.mat是否存在来判断feat_type 特征类型
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode.sh: feature type is $feat_type";

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

#根据feat_type类型，提取的特征处理不一样
case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "$0: Error: Invalid feature type $feat_type" && exit 1;
esac
# 如果没有指定变换输出的文件夹
if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: Error: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)
  #jobs数目 与transform_dir里面的jobs数目不相等，则文档后缀用索引数字
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    echo "$0: num-jobs for transforms mismatches, so copying them."
    for n in $(seq $nj_orig); do cat $transform_dir/trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/trans.ark,$dir/trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
  fi
fi

if [ $stage -le 0 ]; then
  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
      { echo "$0: Error: Mismatch in number of pdfs with $model"; exit 1; }
  fi
  #gmm-latgen-faster解码
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
    gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $decode_extra_opts \
    $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if [ $stage -le 1 ]; then
  [ ! -z $iter ] && iter_opt="--iter $iter"
  steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
fi

if ! $skip_scoring ; then #是否执行评分脚本
  [ ! -x local/score.sh ] && \
    echo "$0: Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $graphdir $dir ||
    { echo "$0: Error: scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
```

### steps/diagnostic/analyze_lats.sh

```shell
#!/usr/bin/env bash
# 此脚本执行的诊断类型与analytics_alignments.sh不同之处在于它从lattices开始（因此必须先将lattices转换为alignments）。
# lattice本质上是一个有向无环图。图上的每个节点代表一个词的结束时间点，每条边代表一个可能的词

# begin configuration section.
iter=final
cmd=run.pl
acwt=0.1
#end configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] (<lang-dir>|<graph-dir>) <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --acwt <acoustic-scale>         # Acoustic scale for getting best-path (default: 0.1)"
  echo "e.g.:"
  echo "$0 data/lang exp/tri4b/decode_dev"
  echo "This script writes some diagnostics to <decode-dir>/log/alignments.log"
  exit 1;
fi

lang=$1 #解码图
dir=$2  #测试输出结果

model=$dir/../${iter}.mdl

for f in $lang/words.txt $model $dir/lat.1.gz $dir/num_jobs; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

num_jobs=$(cat $dir/num_jobs) || exit 1

mkdir -p $dir/log

rm $dir/phone_stats.*.gz 2>/dev/null || true

# this writes two archives of depth_tmp and ali_tmp of (depth per frame, alignment per frame).
# lattice-best-path:通过lattice生成one-best路径，转录或者对齐作为输出。
$cmd JOB=1:$num_jobs $dir/log/lattice_best_path.JOB.log \
  lattice-depth-per-frame "ark:gunzip -c $dir/lat.JOB.gz|" "ark,t:|gzip -c > $dir/depth_tmp.JOB.gz" ark:- \| \
  lattice-best-path --acoustic-scale=$acwt ark:- ark:/dev/null "ark,t:|gzip -c >$dir/ali_tmp.JOB.gz" || exit 1

# ali-to-phones 根据对齐取出音素序列
# ali-to-phones --write-lengths=true exp/chain/tdnn_1a_sp/final.mdl "ark:gunzip -c exp/chain/tdnn_1a_sp/decode_test/ali_tmp.JOB.gz|" ark,t:- \|perl -ne 'chomp;s/^\S+\s*//;@a=split /\s;\s/, $_;$count{"begin ".$a[$0]."\n"}++;if(@a>1){$count{"end ".$a[-1]."\n"}++;}for($i=0;$i<@a;$i++){$count{"all ".$a[$i]."\n"}++;} END{for $k (sort keys %count){print "$count{$k} $k"}}' \|gzip -c '>' $dir/phone_stats.JOB.gz
$cmd JOB=1:$num_jobs $dir/log/get_lattice_stats.JOB.log \
  ali-to-phones --write-lengths=true "$model" "ark:gunzip -c $dir/ali_tmp.JOB.gz|" ark,t:- \| \
  perl -ne 'chomp;s/^\S+\s*//;@a=split /\s;\s/, $_;$count{"begin ".$a[$0]."\n"}++;
  if(@a>1){$count{"end ".$a[-1]."\n"}++;}for($i=0;$i<@a;$i++){$count{"all ".$a[$i]."\n"}++;}
  END{for $k (sort keys %count){print "$count{$k} $k"}}' \| \
  gzip -c '>' $dir/phone_stats.JOB.gz || exit 1

# gunzip -c "exp/chain/tdnn_1a_sp/decode_test/phone_stats.*.gz" | steps/diagnostic/analyze_phone_length_stats.py exp/chain/tdnn_1a_sp/graph
$cmd $dir/log/analyze_alignments.log \
  gunzip -c "$dir/phone_stats.*.gz" \| \
  steps/diagnostic/analyze_phone_length_stats.py $lang || exit 1

grep WARNING $dir/log/analyze_alignments.log
echo "$0: see stats in $dir/log/analyze_alignments.log"


# note: below, some things that would be interpreted by the shell have to be
# escaped since it needs to be passed to $cmd.
# the 'paste' command will paste together the phone-indexes and the depths
# so that one line will be like utt-id1 phone1 phone2 phone3 .. utt-id1 depth1 depth2 depth3 ...
# the following command computes counts of pairs (phone, lattice-depth) and outputs lines
# containing 3 integers representing:
#   phone lattice_depth, count[phone,lattice_depth]
# ali-to-phones --per-frame=true exp/chain/tdnn_1a_sp/final.mdl "ark:gunzip -c $dir/ali_tmp.JOB.gz|" ark,t:- | paste /dev/stdin '<(' gunzip -c $dir/depth_tmp.JOB.gz  ')'  | perl -ane '$half=@F/2;for($i=1;$i<$half;$i++){$j=$i+$half;$count{$F[$i]." ".$F[$j]}++;} END{for $k (sort keys %count){print "$k $count{$k}\n"}}' \| gzip -c '>' exp/chain/tdnn_1a_sp/decode_test/depth_stats_tmp.JOB.gz
$cmd JOB=1:$num_jobs $dir/log/lattice_best_path.JOB.log \
  ali-to-phones --per-frame=true "$model" "ark:gunzip -c $dir/ali_tmp.JOB.gz|" ark,t:- \| \
  paste /dev/stdin '<(' gunzip -c $dir/depth_tmp.JOB.gz  ')'  \| \
  perl -ane '$half=@F/2;for($i=1;$i<$half;$i++){$j=$i+$half;$count{$F[$i]." ".$F[$j]}++;}
  END{for $k (sort keys %count){print "$k $count{$k}\n"}}' \| \
  gzip -c '>' $dir/depth_stats_tmp.JOB.gz

# gunzip -c "exp/chain/tdnn_1a_sp/decode_test/depth_stats_tmp.*.gz" | steps/diagnostic/analyze_lattice_depth_stats.py exp/chain/tdnn_1a_sp/graph
$cmd $dir/log/analyze_lattice_depth_stats.log \
  gunzip -c "$dir/depth_stats_tmp.*.gz" \| \
  steps/diagnostic/analyze_lattice_depth_stats.py $lang || exit 1

grep Overall $dir/log/analyze_lattice_depth_stats.log
echo "$0: see stats in $dir/log/analyze_lattice_depth_stats.log"


rm $dir/phone_stats.*.gz
rm $dir/depth_tmp.*.gz
rm $dir/depth_stats_tmp.*.gz
rm $dir/ali_tmp.*.gz

exit 0
```

### local/score.sh

```shell
# See the script steps/scoring/score_kaldi_cer.sh in case you need to evalutate CER

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=false
stats=true
beam=6
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
iter=final
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1 # 测试数据
lang_or_graph=$2 # 解码图
dir=$3 # 测试结果输出文件夹

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done


ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"


if $decode_mbr ; then
  echo "$0: scoring with MBR, word insertion penalty=$word_ins_penalty"
else
  echo "$0: scoring with word insertion penalty=$word_ins_penalty"
fi


mkdir -p $dir/scoring_kaldi
cat $data/text | $ref_filtering_cmd > $dir/scoring_kaldi/test_filt.txt || exit 1;
if [ $stage -le 0 ]; then

  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    mkdir -p $dir/scoring_kaldi/penalty_$wip/log

    if $decode_mbr ; then
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/best_path.LMWT.log \
        acwt=\`perl -e \"print 1.0/LMWT\"\`\; \
        lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
        lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
        lattice-prune --beam=$beam ark:- ark:- \| \
        lattice-mbr-decode  --word-symbol-table=$symtab \
        ark:- ark,t:- \| \
        utils/int2sym.pl -f 2- $symtab \| \
        $hyp_filtering_cmd '>' $dir/scoring_kaldi/penalty_$wip/LMWT.txt || exit 1;

    else
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/best_path.LMWT.log \
        lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
        lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
        lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
        utils/int2sym.pl -f 2- $symtab \| \
        $hyp_filtering_cmd '>' $dir/scoring_kaldi/penalty_$wip/LMWT.txt || exit 1;
    fi

    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/score.LMWT.log \
      cat $dir/scoring_kaldi/penalty_$wip/LMWT.txt \| \
      compute-wer --text --mode=present \
      ark:$dir/scoring_kaldi/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT_$wip || exit 1;

  done
fi



if [ $stage -le 1 ]; then

  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    for lmwt in $(seq $min_lmwt $max_lmwt); do
      # adding /dev/null to the command list below forces grep to output the filename
      grep WER $dir/wer_${lmwt}_${wip} /dev/null
    done
  done | utils/best_wer.sh  >& $dir/scoring_kaldi/best_wer || exit 1

  best_wer_file=$(awk '{print $NF}' $dir/scoring_kaldi/best_wer)
  best_wip=$(echo $best_wer_file | awk -F_ '{print $NF}')
  best_lmwt=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')

  if [ -z "$best_lmwt" ]; then
    echo "$0: we could not get the details of the best WER from the file $dir/wer_*.  Probably something went wrong."
    exit 1;
  fi

  if $stats; then
    mkdir -p $dir/scoring_kaldi/wer_details
    echo $best_lmwt > $dir/scoring_kaldi/wer_details/lmwt # record best language model weight
    echo $best_wip > $dir/scoring_kaldi/wer_details/wip # record best word insertion penalty

    $cmd $dir/scoring_kaldi/log/stats1.log \
      cat $dir/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt \| \
      align-text --special-symbol="'***'" ark:$dir/scoring_kaldi/test_filt.txt ark:- ark,t:- \|  \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring_kaldi/wer_details/per_utt \|\
       utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring_kaldi/wer_details/per_spk || exit 1;

    $cmd $dir/scoring_kaldi/log/stats2.log \
      cat $dir/scoring_kaldi/wer_details/per_utt \| \
      utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
      sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring_kaldi/wer_details/ops || exit 1;

    $cmd $dir/scoring_kaldi/log/wer_bootci.log \
      compute-wer-bootci --mode=present \
        ark:$dir/scoring_kaldi/test_filt.txt ark:$dir/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt \
        '>' $dir/scoring_kaldi/wer_details/wer_bootci || exit 1;

  fi
fi

# If we got here, the scoring was successful.
# As a  small aid to prevent confusion, we remove all wer_{?,??} files;
# these originate from the previous version of the scoring files
# i keep both statement here because it could lead to confusion about
# the capabilities of the script (we don't do cer in the script)
rm $dir/wer_{?,??} 2>/dev/null
rm $dir/cer_{?,??} 2>/dev/null

exit 0;
```

### local/chain/run_tdnn.sh

```shell
#!/usr/bin/env bash
# This script is based on run_tdnn_7h.sh in swbd chain recipe.
set -e

# configs for 'chain'
affix=
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_1a  # Note: _sp will get added to this
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=12
minibatch_size=128
frames_per_eg=150,110,90
remove_egs=true
common_egs_dir=
xent_regularize=0.1

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
# exp/chain/tdnn_1a_sp
dir=${dir}${affix:+_$affix}_sp
train_set=train_sp
ali_dir=exp/tri5a_sp_ali
treedir=exp/chain/tri6_7d_tree_sp
lang=data/lang_chain


# if we are using the speed-perturbed data we need to generate
# alignments for it.
# 如果我们使用速度扰动的数据，则需要为其生成对齐
local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 7 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  # 将对齐方式获取为网格（给予LF-MMI训练更大的自由度）。使用与对齐相同的num-jobs
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri5a exp/tri5a_sp_lats
  rm exp/tri5a_sp_lats/fsts.*.gz # save space
fi

if [ $stage -le 8 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 9 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  # 使用我们的新拓扑构建一棵树。 与其他配方相比，这是至关重要的步骤。
  # exp/chain/tri6_7d_tree_sp
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 5000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 10 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5
  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 11 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri5a_sp_lats \
    --dir $dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 13 ]; then
  for test_set in dev test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 10 --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_$test_set \
      $graph_dir data/${test_set}_hires $dir/decode_${test_set} || exit 1;
  done
fi
#  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 10 --cmd run.pl --online-ivector-dir exp/nnet3/ivectors_test exp/chain/tdnn_1a/graph data/test_hires exp/chain/tdnn_1a/decode_test
exit;
```

### local/nnet3/run_ivector_common.sh

```shell
#!/usr/bin/env bash

set -euo pipefail

# This script is modified based on mini_librispeech/s5/local/nnet3/run_ivector_common.sh

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=0
train_set=train
test_sets="dev test"
gmm=tri5a
online=false
nnet3_affix=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm_dir=exp/${gmm}  # gmm模型文件夹
ali_dir=exp/${gmm}_sp_ali  #对齐后文件夹
# 判断数据文件和模型文件是否存在
for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

online_affix=
if [ $online = true ]; then
  online_affix=_online
fi

if [ $stage -le 1 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  # 尽管nnet将由高分辨率数据训练，但我们仍然必须干扰正常数据才能获得对齐方式_sp代表速度干扰
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 70 data/${train_set}_sp \
    exp/make_mfcc/train_sp mfcc_perturbed || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp \
    exp/make_mfcc/train_sp mfcc_perturbed || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  # 与扰动的低分辨率数据对齐
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  # 创建高分辨率MFCC功能（使用40个倒谱而不是13个）。这显示了如何拆分多个文件系统
  echo "$0: creating high-resolution MFCC features"
  mfccdir=mfcc_perturbed_hires$online_affix
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/aishell-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires$online_affix
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  # 在提取招聘特征之前对培训数据进行体积微扰； 这有助于使训练有素的nnet更加不变地测试数据量。
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires$online_affix || exit 1;

  for datadir in ${train_set}_sp ${test_sets}; do
    steps/make_mfcc_pitch$online_affix.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires$online_affix exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires$online_affix exp/make_hires/$datadir $mfccdir || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires$online_affix || exit 1;
    # create MFCC data dir without pitch to extract iVector
    utils/data/limit_feature_dim.sh 0:39 data/${datadir}_hires$online_affix data/${datadir}_hires_nopitch || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires_nopitch exp/make_hires/$datadir $mfccdir || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."
  # We'll use about a quarter of the data.
  # 我们将使用大约四分之一的数据。
  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

  num_utts_total=$(wc -l <data/${train_set}_sp_hires_nopitch/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires_nopitch \
     $num_utts ${temp_data_root}/${train_set}_sp_hires_nopitch_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       ${temp_data_root}/${train_set}_sp_hires_nopitch_subset \
       exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads 8 \
    ${temp_data_root}/${train_set}_sp_hires_nopitch_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 5 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
     data/${train_set}_sp_hires_nopitch exp/nnet3${nnet3_affix}/diag_ubm \
     exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

train_set=train_sp

if [ $stage -le 6 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.
  # 在组合了短片段之后，我们在速度受扰的训练数据上提取了iVectors，这将成为我们训练系统的基础。 使用--utts-per-spk-max 2，脚本会将发话配对成两半，并将每对发话当作一个说话者。 这将使iVectors更具多样性。请注意，这些是“在线”提取的。

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.
  # 注意，即使那是我们从中提取ivector的数据，我们也不会在ivectordir的名称中对'max2'进行编码，因为对于非'max2'数据仍然有效，发声列表是相同的 。

  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/aishell-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  # 拥有更多的发言者有助于泛化，并能很好地处理每个话语的解码（iVector从零开始）。
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_hires_nopitch ${temp_data_root}/${train_set}_sp_hires_nopitch_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    ${temp_data_root}/${train_set}_sp_hires_nopitch_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      data/${data}_hires_nopitch exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}
  done
fi

exit 0
```

### steps/nnet3/decode.sh

```sh
#!/usr/bin/env bash
# This script does decoding with a neural-net.
# Begin configuration section.
stage=1
nj=4 # number of decoding jobs.
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=1.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.
cmd=run.pl
beam=15.0
frames_per_chunk=50
max_active=7000
min_active=200
ivector_scale=1.0
lattice_beam=8.0 # Beam we use in lattice generation.
iter=final
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
use_gpu=false # If true, will use a GPU, with nnet3-latgen-faster-batch.
              # In that case it is recommended to set num-threads to a large
              # number, e.g. 20 if you have that many free CPU slots on a GPU
              # node, and to use a small number of jobs.
scoring_opts=
skip_diagnostics=false
skip_scoring=false
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=     #exp/nnet3/ivectors_test
minimize=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
  echo "e.g.:   steps/nnet3/decode.sh --nj 8 \\"
  echo "--online-ivector-dir exp/nnet2_online/ivectors_test_eval92 \\"
  echo "    exp/tri4b/graph_bg data/test_eval92_hires $dir/decode_bg_eval92"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 15.0"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  echo "  --num-threads <n>                        # number of threads to use, default 1."
  echo "  --use-gpu <true|false>                   # default: false.  If true, we recommend"
  echo "                                           # to use large --num-threads as the graph"
  echo "                                           # search becomes the limiting factor."
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
model=$srcdir/$iter.mdl

extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

utils/lang/check_phones_compatible.sh {$srcdir,$graphdir}/phones.txt || exit 1

for f in $graphdir/HCLG.fst $data/feats.scp $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
if [ -f $srcdir/cmvn_opts ]; then
    cmvn_opts=`cat $srcdir/cmvn_opts` 
else
    cmvn_opts="--norm-means=false --norm-vars=false"
fi
thread_string=
if $use_gpu; then
  if [ $num_threads -eq 1 ]; then
    echo "$0: **Warning: we recommend to use --num-threads > 1 for GPU-based decoding."
  fi
  thread_string="-batch --num-threads=$num_threads"
  queue_opt="--num-threads $num_threads --gpu 1"
elif [ $num_threads -gt 1 ]; then
  thread_string="-parallel --num-threads=$num_threads"
  queue_opt="--num-threads $num_threads"
fi

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

## Set up features.
if [ -f $srcdir/online_cmvn ]; then online_cmvn=true
else online_cmvn=false; fi

if ! $online_cmvn; then
echo "$0: feature type is raw"
  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
else
  feats="ark,s,cs:apply-cmvn-online $cmvn_opts --spk2utt=ark:$sdata/JOB/spk2utt $srcdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- |"
fi

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
elif [ -f $srcdir/init/info.txt ]; then
    frame_subsampling_factor=$(awk '/^frame_subsampling_factor/ {print $2}' <$srcdir/init/info.txt)
    if [ ! -z $frame_subsampling_factor ]; then
        frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
    fi
fi

if [ $stage -le 1 ]; then
# nnet3-latgen-faster --online-ivectors=scp:exp/nnet3/ivector_test/ivector_online.scp --online-ivector-period=10 --frame_subsampling_factor=3 --frames-per-chunk=50 --extra-left-context=0 --extra-right-context=0 --extra-left-context-initial=-1 --extra-right-context-final=-1 --minimize=false --max-active=7000 --min-active=200 --beam=15.0 --lattice-beam=8.0 --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=exp/chain/tdnn_1a_sp/graph/words.txt "exp/chain/tdnn_1a_sp/final.mdl" exp/chain/tdnn_1a_sp/graph/HCLG.fst "ark,s,cs:apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/test_hires/split10/JOB/utt2spk scp:data/test_hires/split10/JOB/cmvn.scp scp:data/test_hires/split10/JOB/feats.scp ark:- |" "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c > exp/chain/tdnn_1a_sp/decode_test/lat.JOB.gz"
  $cmd $queue_opt JOB=1:$nj $dir/log/decode.JOB.log \
    nnet3-latgen-faster$thread_string $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam \
     --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
     --word-symbol-table=$graphdir/words.txt "$model" \
     $graphdir/HCLG.fst "$feats" "$lat_wspecifier" || exit 1;
fi


if [ $stage -le 2 ]; then
  if ! $skip_diagnostics ; then
    [ ! -z $iter ] && iter_opt="--iter $iter"
    # steps/diagnostic/analyze_lats.sh --cmd run.pl --iter final exp/chain/tdnn_1a_sp/graph exp/chain/tdnn_1a_sp/decode_test
    steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
  fi
fi


# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.
if [ $stage -le 3 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
    # local/score.sh --cmd run.pl data/test_hires/ exp/chain/tdnn_1a_sp/graph exp/chain/tdnn_1a_sp/decode_test
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi
echo "Decoding done."
exit 0;
```









## bin

latgen-faster-mapped：生成lattice
lattice-scale：对lattice权重缩放。
lattice-add-penalty:lattice中添加单词插入惩罚
lattice-best-path:通过lattice生成one-best路径，转录或者对齐作为输出。
lattice-align-words：转换lattice，使CompactLattice格式的弧与字词对应。
lattice-align-phones：转换lattices，使CompactLattice格式的弧与音素对应。



