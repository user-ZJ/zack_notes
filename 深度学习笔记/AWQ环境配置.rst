AWQ环境配置
======================

基础镜像
----------------------
pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

AutoAWQ安装
----------------------
1. 运行容器

.. code-block:: bash

    docker run --gpus all --name awq -v /nasdata/zhanjie/models/:/models -it pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel /bin/bash

2. 配置基础环境

.. code-block:: bash

    apt update
    apt install git wget unzip

3. 安装AutoAWQ

.. code-block:: bash

    wget https://github.com/casper-hansen/AutoAWQ_kernels/releases/download/v0.0.2/autoawq_kernels-0.0.2+cu118-cp310-cp310-linux_x86_64.whl
    wget https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.8/autoawq-0.1.8+cu118-cp310-cp310-linux_x86_64.whl
    pip install autoawq_kernels-0.0.2+cu118-cp310-cp310-linux_x86_64.whl
    pip install autoawq-0.1.8+cu118-cp310-cp310-linux_x86_64.whl
    pip install sentencepiece

量化llama-7b模型
----------------------
awq使用的是部分mit-han-lab数据集进行校验

.. code-block:: python 

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer,TextStreamer


    def build_prompt(prompt,history=[]):
        sysinfo = ''
        prompt = f"Human: {prompt} \n\nAssistant: "
        if history:
            temp = ''
            for query, res in history:
                temp += f"Human: {query}\nAssistant: {res}\n"
            prompt = temp + prompt
        prompt = sysinfo + prompt
        return prompt

    model_path = '/models/belle-llama-7b-2m/'
    quant_path = 'llama-7b-awq-4bit'
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,use_fast=False)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    del model
    del tokenizer
    model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
    tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True,use_fast=False)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    query = "药物治疗能好吗？ [回复尽量限制50字以内]"
    history = [["得了抑郁症怎么办？","建议寻求专业心理医生的帮助，进行药物治疗和心理疏导。同时，保持积极乐观的心态，多与亲友交流，参加一些有益身心健康的活动。"]]

    prompt = build_prompt(query,history)
    print(prompt)

    input_tokens = tokenizer(prompt, return_tensors='pt').input_ids
    input_tokens = input_tokens.to('cuda:0')
    print(input_tokens)
    pred = model.generate(input_tokens,temperature=0.1,do_sample=True,top_p=0.75,repetition_penalty=1.2,max_new_tokens=2048)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    print("response",response)



Q&A
----------------------
校验数据下载失败
`````````````````````````
1. 校验数据集为https://huggingface.co/datasets/mit-han-lab/pile-val-backup/tree/main
2. 下载数据val.jsonl.zst
3. 装置zstd解压软件，apt install zstd
4. 解压zstd -d val.jsonl.zst
5. 将解压后的val.jsonl复制到容器中的/workspace目录
6. 修改数据加载代码。修改/workspace/AutoAWQ/awq/utils/calib_data.py第12行为

.. code-block:: python

    dataset = load_dataset("json",data_files={"validation":"/workspace/val.jsonl"}, split="validation")

