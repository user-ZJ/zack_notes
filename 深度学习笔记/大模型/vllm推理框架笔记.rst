vllm推理框架笔记
==============================

启动命令：

CUDA_VISIBLE_DEVICES=1,3 python -m vllm.entrypoints.openai.api_server --model /models/glm-4-9b-chat/ --enable-lora --lora-modules cnjy-cut-glm4-lora=/models/glm-4-9b-checkpoint-25000 --served-model-name glm-4-9b-chat --trust-remote-code --tensor-parallel-size 2 --distributed-executor-backend=ray --engine-use-ray --enforce-eager --dtype bfloat16 --max-model-len 16000 --disable-custom-all-reduce --port 80

CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model /models/glm-4-9b-chat/ --served-model-name glm-4-9b-chat --trust-remote-code --tensor-parallel-size 2 --distributed-executor-backend=ray --engine-use-ray --enforce-eager --dtype bfloat16 --max-model-len 16000 --disable-custom-all-reduce --port 80
