# kaldi xvector训练

kaldi中xvector训练脚本有两个：

https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16/v2

https://github.com/kaldi-asr/kaldi/tree/master/egs/voxceleb/v2

其中sre16数据集不公开，所以以voxceleb中脚本来说明xvector训练过程

## 数据准备

下载[voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)数据集，解压后整理格式如下：

```
VoxCeleb1/dev/wav
VoxCeleb1/test/wav
```

下载[voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)数据集，解压后整理格式如下：

```
VoxCeleb2/dev/aac
VoxCeleb2/test/aac
```

下载[musan](https://www.openslr.org/17/)数据集，解压

下载[rirs_noises](https://www.openslr.org/28/)数据集，加压到egs/voxceleb/v2目录下

修改run.sh脚本中的：

```shell
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
nnet_dir=exp/xvector_nnet_1a
musan_root=/export/corpora/JHU/musan
```

为：

```shell
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/path/to/your/VoxCeleb1
voxceleb2_root=/path/to/your/VoxCeleb2
nnet_dir=exp/xvector_nnet_1a
musan_root=/path/to/your/musan
```



