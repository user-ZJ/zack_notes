## 数据处理

1. 数据预处理

```shell
local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
数据整理
1. 整理所有wav文件路径到wav.flist
2. 整理训练wav文件路径到data/local/train/wav.flist,验证数据到data/local/dev/wav.flist,测试数据到data/local/test/wav.flist
3. 准备训练，验证，测试集的  utt2spk，wav.scp，text，spk2utt
```

2. 特征提取

```shell
//计算wav的fbank特征和pitch特征，并将fbank特征和pitch特征进行连接
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
    data/train exp/make_fbank/train fbank
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/dev exp/make_fbank/dev fbank
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/test exp/make_fbank/test fbank
//速度扰动
utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
utils/combine_data.sh --extra-files utt2uniq data/train_sp data/temp1 data/temp2 data/temp3
rm -r data/temp1 data/temp2 data/temp3
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        data/train_sp exp/make_fbank/train_sp fbank
//计算归一化参数
compute-cmvn-stats scp:data/train_sp/feats.scp data/train_sp/cmvn.ark
//对特征进行归一化（包括训练，验证和测试集）
dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/train_sp/feats.scp data/train_sp/cmvn.ark exp/dump_feats/train dump/train_sp/deltafalse
```

3. 生成json文件，用于训练和测试

```shell
data2json.sh --feat dump/train_sp/deltafalse/feats.scp \
data/train_sp data/lang_1char/train_sp_units.txt > dump/train_sp/deltafalse/data.json
data.json为特征，文本，文本id对应关系
"""
input 输入特征和shape(Nx83)
output 输出text,token,tokenid,shape(Nx4233)
"""
1. feat_to_shape.sh --cmd run.pl --nj 1 --filetype  --preprocess-conf  --verbose 0 dump/train_sp/deltafalse/feats.scp data/train_sp/tmp-kaLnz/input_1/shape.scp
2. text2token.py -s 1 -n 1 data/train_sp/text --trans_type char > data/train_sp/tmp-GMET8/output/token.scp
3. < ${tmpdir}/output/token.scp utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/output/tokenid.scp
4. < data/train_sp/tmp-3z70N/output/token.scp utils/sym2int.pl --map-oov <unk> -f 2- data/lang_1char/train_sp_units.txt > data/train_sp/tmp-3z70N/output/tokenid.scp
5. < data/train_sp/tmp-3z70N/output/token.scp utils/sym2int.pl --map-oov <unk> -f 2- data/lang_1char/train_sp_units.txt > data/train_sp/tmp-3z70N/output/tokenid.scp
6. merge_scp2json.py --verbose 0 --input-scps feat:data/train_sp/tmp-kaLnz/input_1/feat.scp shape:data/train_sp/tmp-kaLnz/input_1/shape.scp:shape --output-scps shape:data/train_sp/tmp-kaLnz/output/shape.scp:shape text:data/train_sp/tmp-kaLnz/output/text.scp token:data/train_sp/tmp-kaLnz/output/token.scp tokenid:data/train_sp/tmp-kaLnz/output/tokenid.scp --scps utt2spk:data/train_sp/tmp-kaLnz/other/utt2spk.scp --allow-one-column false
```

## 语言模型训练

```shell
lm_train.py \
--config conf/lm.yaml \
--ngpu 1 \
--backend pytorch \
--verbose 1 \
--outdir exp/train_rnnlm_pytorch_lm \
--tensorboard-dir tensorboard/train_rnnlm_pytorch_lm \
--train-label data/local/lm_train/train.txt \
--valid-label data/local/lm_train/valid.txt \
--resume  \
--dict data/lang_1char/train_sp_units.txt
```

## 声学模型训练

```shell
asr_train.py \
--config ${train_config} \
--ngpu ${ngpu} \
--backend ${backend} \
--outdir ${expdir}/results \
--tensorboard-dir tensorboard/${expname} \
--debugmode ${debugmode} \
--dict ${dict} \
--debugdir ${expdir} \
--minibatches ${N} \
--verbose ${verbose} \
--resume ${resume} \
--train-json ${feat_tr_dir}/data.json \
--valid-json ${feat_dt_dir}/data.json
```

## 识别

```
asr_recog.py \
--config conf/decode.yaml \
--ngpu 0 \
--backend pytorch \
--batchsize 0 \
--recog-json dump/test/deltafalse/split32utt/data.1.json \
--result-label exp/train_sp_pytorch_train/decode_test_decode_lm/data.1.json \
--model exp/train_sp_pytorch_train/results/model.last10.avg.best  \
--rnnlm exp/train_rnnlm_pytorch_lm/rnnlm.model.best
score_sclite.sh ${expdir}/${decode_dir} ${dict}
```

```python
from espnet.asr.pytorch_backend.asr import recog
recog(args)
model, train_args = load_trained_model(args.model)
rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
feat = load_inputs_and_targets(batch)[0][0]
nbest_hyps = model.recognize(
                        feat, args, train_args.char_list, rnnlm
                    )
rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)
# 提取hyp["yseq"][1:]和hyp["score"]两个字段
```



<details><summary>args/recog_args:</summary><div>     

```
Namespace(
api='v1', 
backend='pytorch', 
batchsize=0, 
beam_size=10, 
config='conf/decode.yaml', 
config2=None, 
config3=None, 
ctc_weight=0.5, 
ctc_window_margin=0, 
debugmode=1, 
dtype='float32', 
lm_weight=0.7, 
maxlenratio=0.0, 
minlenratio=0.0, 
model='exp/train_sp_pytorch_train/results/model.last10.avg.best', 
model_conf=None, 
nbest=1, 
ngpu=0, 
num_encs=1, 
num_spkrs=1, 
penalty=0.0, 
preprocess_conf=None, 
recog_json='dump/test/deltafalse/split32utt/data.1.json', 
result_label='exp/train_sp_pytorch_train/decode_test_decode_lm/data.1.json', 
rnnlm='exp/train_rnnlm_pytorch_lm/rnnlm.model.best', 
rnnlm_conf=None, 
score_norm_transducer=True, 
seed=1, 
streaming_min_blank_dur=10, 
streaming_mode=None, 
streaming_offset_margin=1, 
streaming_onset_margin=1, 
streaming_window=10, verbose=1, 
weights_ctc_dec=None, 
word_dict=None, 
word_rnnlm=None, 
word_rnnlm_conf=None)
```

</div></details>

<details><summary>train_args:</summary><div> 

```
Namespace(
accum_grad=2, 
adim=256, 
aheads=4, 
apply_uttmvn=True, 
backend='pytorch',
badim=320, 
batch_bins=0,
batch_count='auto', 
batch_frames_in=0, 
batch_frames_inout=0, 
batch_frames_out=0,
batch_size=32, 
bdropout_rate=0.0, 
beam_size=4, 
blayers=2, 
bnmask=2, 
bprojs=300, 
btype='blstmp', 
bunits=300, 
char_list=['<blank>', '<unk>', '一', '丁', '七',..., '<eos>'],
config='conf/train.yaml', 
config2=None, 
config3=None, 
context_residual=False,
criterion='acc', 
ctc_type='warpctc',
ctc_weight=0.3, 
debugdir='exp/train_sp_pytorch_train', 
debugmode=1, 
dec_init=None, 
dec_init_mods=['att.', ' dec.'], 
dict='data/lang_1char/train_sp_units.txt', 
dlayers=6, 
dropout_rate=0.1, 
dunits=2048, 
early_stop_criterion='validation/main/acc', 
elayers=12, 
enc_init=None, 
enc_init_mods=['enc.enc.'], 
epochs=50, 
eps=1e-08, 
eps_decay=0.01, 
eunits=2048, 
fbank_fmax=None,
fbank_fmin=0.0, 
fbank_fs=16000, 
grad_clip=5.0, 
grad_noise=False,
lm_weight=0.1, 
lsm_weight=0.1,
maxlen_in=512,
maxlen_out=150, 
maxlenratio=0.0, 
minibatches=0,
minlenratio=0.0,
model_module='espnet.nets.pytorch_backend.e2e_asr_transformer:E2E', 
mtlalpha=0.3,
n_iter_processes=0, 
n_mels=80, 
nbest=1, 
ngpu=1, 
num_encs=1,
num_save_attention=3,
num_spkrs=1, 
opt='noam', 
outdir='exp/train_sp_pytorch_train/results', 
patience=0, 
penalty=0.0, 
preprocess_conf=None, 
ref_channel=-1, 
report_cer=False, 
report_interval_iters=100,
report_wer=False, 
resume=None, 
rnnlm=None, 
rnnlm_conf=None, 
save_interval_iters=0, 
seed=1, 
sortagrad=0, 
stats_file=None, 
sym_blank='<blank>', 
sym_space='<space>', 
tensorboard_dir='tensorboard/train_sp_pytorch_train', 
threshold=0.0001, 
train_dtype='float32', 
train_json='dump/train_sp/deltafalse/data.json',
transformer_attn_dropout_rate=0.0, 
transformer_init='pytorch', 
transformer_input_layer='conv2d', 
transformer_length_normalized_loss=0, 
transformer_lr=1.0, 
transformer_warmup_steps=25000, 
use_beamformer=True, 
use_dnn_mask_for_wpe=False, 
use_frontend=False, 
use_wpe=False, 
uttmvn_norm_means=True, 
uttmvn_norm_vars=False, 
valid_json='dump/dev/deltafalse/data.json', 
verbose=0, 
wdropout_rate=0.0, 
weight_decay=0.0, 
wlayers=2, 
wpe_delay=3, 
wpe_taps=5, 
wprojs=300, 
wtype='blstmp', 
wunits=300)
```



</div></details>

<details><summary>rnnlm_args:</summary><div> 

```
Namespace(
accum_grad=1, 
backend='pytorch',
batchsize=64, 
char_list_dict={'<blank>': 0, '<eos>': 4232, '<unk>': 1, '一': 2, '丁': 3, ...},
config='conf/lm.yaml', 
config2=None, 
config3=None, 
debugmode=1, 
dict='data/lang_1char/train_sp_units.txt', 
dropout_rate=0.5, 
dump_hdf5_path=None, 
early_stop_criterion='validation/main/loss', 
embed_unit=None, 
epoch=20, 
gradclip=5, 
layer=2, 
lr=1.0, 
maxlen=100, 
model_module='default', 
n_vocab=4233, 
ngpu=0, 
opt='sgd', 
outdir='exp/train_rnnlm_pytorch_lm', 
patience=3, 
report_interval_iters=100, 
resume=None, 
schedulers=None, 
seed=1, 
sortagrad=0, 
tensorboard_dir='tensorboard/train_rnnlm_pytorch_lm', 
test_label=None, train_dtype='float32', 
train_label='data/local/lm_train/train.txt', 
type='lstm', 
unit=650, 
valid_label='data/local/lm_train/valid.txt', 
verbose=1, 
weight_decay=0.0)
```



</div></details>










