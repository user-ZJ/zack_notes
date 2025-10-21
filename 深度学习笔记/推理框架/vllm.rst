vllm
============

V1
-------------------
https://zhuanlan.zhihu.com/p/1900126076279160869

.. mermaid::

    sequenceDiagram
        participant OpenAIServingChat
        participant AsyncLLM
        participant Processor
        participant InputPreprocessor
        participant Qwen3OmniMoeThinkerMultiModalProcessor
        participant EngineCore
        participant Scheduler
        participant UniProcExecutor
        participant Worker(gpu_worker)
        participant GPUModelRunner
        participant XXXXModel


        OpenAIServingChat->>OpenAIServingChat: create_chat_completion
        OpenAIServingChat->>OpenAIServingChat: _preprocess_chat #读取语音和图片
        OpenAIServingChat->>AsyncLLM: generate
        AsyncLLM->>AsyncLLM: add_request
        AsyncLLM->>Processor: process_inputs
        Processor->>InputPreprocessor: preprocess
        InputPreprocessor->>InputPreprocessor: _process_decoder_only_prompt
        InputPreprocessor->>InputPreprocessor: _prompt_to_llm_inputs
        InputPreprocessor->>InputPreprocessor: _process_tokens
        InputPreprocessor->>InputPreprocessor: _process_multimodal
        InputPreprocessor->>Qwen3OmniMoeThinkerMultiModalProcessor: apply
        AsyncLLM->>AsyncLLM: _add_request
        AsyncLLM->>EngineCore: add_request_async


        EngineCore->>EngineCore: _process_input_queue
        EngineCore->>EngineCore: _handle_client_request
        EngineCore->>EngineCore: add_request
        EngineCore->>Scheduler: add_request
        Scheduler->>Scheduler: self.waiting.append(seq_group)

        EngineCore->>EngineCore: _process_engine_step
        EngineCore->>EngineCore: step
        EngineCore->>Scheduler: schedule
        Scheduler->>Scheduler: _try_schedule_encoder_inputs

        Scheduler->>Scheduler: _make_cached_request_data
        Scheduler->>Scheduler: _update_after_schedule


        EngineCore->>UniProcExecutor: execute_model
        UniProcExecutor->>UniProcExecutor: collective_rpc
        UniProcExecutor->>Worker(gpu_worker): execute_model
        Worker(gpu_worker)->>GPUModelRunner: execute_model
        GPUModelRunner->>GPUModelRunner: _prepare_inputs
        GPUModelRunner->>GPUModelRunner: _execute_mm_encoder
        GPUModelRunner->>XXXXModel: get_multimodal_embeddings
        GPUModelRunner->>XXXXModel: forward
        GPUModelRunner->>XXXXModel: compute_logits
