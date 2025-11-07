vllm
============

V1
-------------------
vllm 0.11.1

https://zhuanlan.zhihu.com/p/1900126076279160869

+--------------------------------+-----------------------------------------+
|               类               |                  文件                   |
+================================+=========================================+
| OpenAIServingChat              | vllm/entrypoints/openai/serving_chat.py |
+--------------------------------+-----------------------------------------+
| AsyncLLM                       | vllm/v1/engine/async_llm.py             |
+--------------------------------+-----------------------------------------+
| Processor                      | vllm/v1/engine/processor.py             |
+--------------------------------+-----------------------------------------+
| InputPreprocessor              | vllm/inpus/preprocessor.py              |
+--------------------------------+-----------------------------------------+
| EngineCore                     | vllm/v1/engine/core.py                  |
+--------------------------------+-----------------------------------------+
| EngineCoreClient;AsyncMPClient | vllm/v1/engine/core_client.py           |
+--------------------------------+-----------------------------------------+
| Scheduler                      | vllm/v1/core/sched/scheduler.py         |
+--------------------------------+-----------------------------------------+
| UniProcExecutor                | vllm/executor/uniproc_executor.py       |
+--------------------------------+-----------------------------------------+
| Worker                         | vllm/v1/worker/gpu_worker.py            |
+--------------------------------+-----------------------------------------+
| GPUModelRunner                 | vllm/v1/worker/gpu_model_runner.py      |
+--------------------------------+-----------------------------------------+
| OutputProcessor                | vllm/v1/engine/output_processor.py      |
+--------------------------------+-----------------------------------------+




.. mermaid::

    sequenceDiagram
        participant api_server.py
        participant OpenAIServingChat
        participant AsyncLLM
        participant EngineCoreClient
        participant AsyncMPClient
        participant EngineCoreProc
        participant Processor
        participant InputPreprocessor
        participant OutputProcessor
        participant QwenXXXMultiModalProcessor as XXXMultiModalProcessor<br/>BaseMultiModalProcessor
        participant Scheduler
        participant UniProcExecutor
        participant Worker(gpu_worker)
        participant GPUModelRunner
        participant XXXXModel

         Note over api_server.py,XXXXModel: 启动流程
        api_server.py->>api_server.py: run_server
        api_server.py->>api_server.py: run_server_worker
        api_server.py->>api_server.py: build_async_engine_client
        api_server.py->>AsyncLLM: from_vllm_config
        AsyncLLM->>Processor: __init__
        AsyncLLM->>OutputProcessor: __init__
        AsyncLLM->>EngineCoreClient: make_sync_mp_client
        EngineCoreClient->>AsyncMPClient: __init__
        AsyncMPClient->>AsyncMPClient: MPClient.__init__ #加载模型
        AsyncMPClient->>AsyncMPClient: MPClient.launch_core_engines #加载模型
        AsyncMPClient->>+EngineCoreProc: run_engine_core
        EngineCoreProc->>+EngineCoreProc: __init__
        EngineCoreProc->>+UniProcExecutor: __init__
        UniProcExecutor->>+Worker(gpu_worker): init_worker
        Worker(gpu_worker)-->>-UniProcExecutor: return
        UniProcExecutor->>+Worker(gpu_worker): init_device
        Worker(gpu_worker)->>+GPUModelRunner: __init__
        GPUModelRunner-->>-Worker(gpu_worker): return
        Worker(gpu_worker)-->>-UniProcExecutor: return
        UniProcExecutor->>+Worker(gpu_worker): load_model
        Worker(gpu_worker)->>+GPUModelRunner: load_model
        GPUModelRunner->>+XXXXModel: __init__
        XXXXModel-->>-GPUModelRunner: return
        GPUModelRunner-->>-Worker(gpu_worker): return
        Worker(gpu_worker)-->>-UniProcExecutor: return
        UniProcExecutor-->>-EngineCoreProc: return
        EngineCoreProc->>+Scheduler: __init__
        Scheduler-->>-EngineCoreProc: return
        EngineCoreProc->>-EngineCoreProc: return
        EngineCoreProc->>+EngineCoreProc: run_busy_loop
        EngineCoreProc->>-EngineCoreProc: return
        EngineCoreProc-->>-AsyncMPClient: return
        AsyncMPClient-->>EngineCoreClient: MPClient
        EngineCoreClient-->>AsyncLLM: engine_core
        AsyncLLM-->>api_server.py: engine_client
        AsyncLLM->>AsyncLLM: _run_output_hanlder
        api_server.py->>api_server.py: build_app
        api_server.py->>api_server.py: init_app_state
        api_server.py->>OpenAIServingChat: __init__

        Note over api_server.py,XXXXModel: 生成阶段 #分割线
        # 生成流程
        OpenAIServingChat->>OpenAIServingChat: create_chat_completion
        OpenAIServingChat->>OpenAIServingChat: _preprocess_chat #读取语音和图片
        OpenAIServingChat->>OpenAIServingChat: _process_inputs
        OpenAIServingChat->>AsyncLLM: generate
        AsyncLLM->>AsyncLLM: add_request
        AsyncLLM->>+Processor: process_inputs
        Processor->>+InputPreprocessor: preprocess
        InputPreprocessor->>+InputPreprocessor: _process_decoder_only_prompt
        InputPreprocessor->>+InputPreprocessor: _prompt_to_llm_inputs
        InputPreprocessor->>+InputPreprocessor: _process_tokens
        InputPreprocessor->>+InputPreprocessor: _process_multimodal
        InputPreprocessor->>+QwenXXXMultiModalProcessor: apply
        QwenXXXMultiModalProcessor->>+QwenXXXMultiModalProcessor: _to_mm_items
        QwenXXXMultiModalProcessor-->>-QwenXXXMultiModalProcessor: return
        QwenXXXMultiModalProcessor->>+QwenXXXMultiModalProcessor: _cache_apply_hf_processor
        QwenXXXMultiModalProcessor-->>-QwenXXXMultiModalProcessor: return
        QwenXXXMultiModalProcessor->>+QwenXXXMultiModalProcessor: _maybe_apply_prompt_updates
        QwenXXXMultiModalProcessor-->>-QwenXXXMultiModalProcessor: return
        QwenXXXMultiModalProcessor-->>-InputPreprocessor: MultiModalInputs
        InputPreprocessor->>-InputPreprocessor: return
        InputPreprocessor->>-InputPreprocessor: return
        InputPreprocessor->>-InputPreprocessor: return
        InputPreprocessor->>-InputPreprocessor: return
        InputPreprocessor-->>-Processor: ProcessorInputs
        Processor-->>-AsyncLLM: EngineCoreRequest
        
        AsyncLLM->>AsyncLLM: _add_request
        AsyncLLM->>AsyncMPClient: add_request_async
        AsyncMPClient->>AsyncMPClient: _send_input

        loop 接收request,并放到input队列
            EngineCoreProc->>EngineCoreProc: process_input_sockets
        end
        loop
            EngineCoreProc->>+EngineCoreProc: run_busy_loop
            EngineCoreProc->>+EngineCoreProc: _process_input_queue
            EngineCoreProc->>+EngineCoreProc: _handle_client_request
            EngineCoreProc->>+EngineCoreProc: add_request
            EngineCoreProc->>Scheduler: add_request
            Scheduler->>Scheduler: self.waiting.append(seq_group)
            EngineCoreProc->>-EngineCoreProc: return
            EngineCoreProc->>-EngineCoreProc: return
            EngineCoreProc->>-EngineCoreProc: return

            EngineCoreProc->>+EngineCoreProc: _process_engine_step
            EngineCoreProc->>+EngineCoreProc: step
            EngineCoreProc->>+Scheduler: schedule
            Scheduler->>Scheduler: _try_schedule_encoder_inputs

            Scheduler->>Scheduler: _make_cached_request_data
            Scheduler->>Scheduler: _update_after_schedule
            Scheduler-->>-EngineCoreProc: scheduler_output


            EngineCoreProc->>+UniProcExecutor: execute_model
            UniProcExecutor->>UniProcExecutor: collective_rpc
            UniProcExecutor->>Worker(gpu_worker): execute_model
            Worker(gpu_worker)->>GPUModelRunner: execute_model
            GPUModelRunner->>GPUModelRunner: _prepare_inputs
            GPUModelRunner->>GPUModelRunner: _execute_mm_encoder
            GPUModelRunner->>XXXXModel: get_multimodal_embeddings
            GPUModelRunner->>XXXXModel: forward
            GPUModelRunner->>XXXXModel: compute_logits
            GPUModelRunner-->>Worker(gpu_worker): model_output
            Worker(gpu_worker)-->>UniProcExecutor: model_output
            UniProcExecutor-->>-EngineCoreProc: model_output

            EngineCoreProc->>+Scheduler: update_from_output
            Scheduler-->>-EngineCoreProc: engine_core_outputs
            EngineCoreProc->>-EngineCoreProc: return
            EngineCoreProc->>-EngineCoreProc: return
            EngineCoreProc->>-EngineCoreProc: return
        end

        loop 处理output队列,并发送数据
            EngineCoreProc->>EngineCoreProc: process_output_sockets_ThreadLoop
        end

