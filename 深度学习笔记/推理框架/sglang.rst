sglang
============

.. mermaid::

    sequenceDiagram
        participant Engine
        participant Scheduler
        participant TpModelWorker
        participant ModelRunner
        participant Sampler
        participant DetokenizerManager
        participant TokenizerManager
        

        Engine->>Engine: _launch_subprocesses
        Engine->>Scheduler: run_scheduler_process
        Scheduler->>TpModelWorker: new TpModelWorker
        TpModelWorker->>ModelRunner: new ModelRunner
        ModelRunner->>Sampler: new Sampler
        ModelRunner->>ModelRunner:load_model
        Engine->>DetokenizerManager: run_detokenizer_process
        Engine->>TokenizerManager: new TokenizerManager



rpc_ipc_name： engine和scheduler进行通行    
scheduler_input_ipc_name: tokenizer发送数据到scheduler  
tokenizer_ipc_name： detokenizer和scheduler发送数据到tokenizer
detokenizer_ipc_name: scheduler发送数据到detokenizer

.. mermaid::

    sequenceDiagram
        participant Engine
        participant TokenizerManager
        participant Scheduler
        participant TpModelWorker
        participant ModelRunner
        participant SchedulerOutputProcessorMixin
        participant DetokenizerManager

        Engine->>Engine:async_generate
        Engine->>TokenizerManager: generate_request
        TokenizerManager->>TokenizerManager: auto_create_handle_loop
        TokenizerManager->>TokenizerManager: _tokenize_one_request
        TokenizerManager->>Scheduler: _send_one_request
        
        loop event_loop_normal
            Scheduler->>Scheduler: recv_requests

            activate Scheduler
            Scheduler->>Scheduler: process_input_requests
            Scheduler->>Scheduler: handle_generate_request
            Scheduler->>Scheduler: _add_request_to_queue
            deactivate Scheduler

            Scheduler->>Scheduler: get_next_batch_to_run

            activate Scheduler
            Scheduler->>Scheduler: run_batch
            Scheduler->>Scheduler: get_model_worker_batch
            activate TpModelWorker
            Scheduler->>TpModelWorker: forward_batch_generation
            TpModelWorker->>ModelRunner: forward
            TpModelWorker->>ModelRunner: sample
            TpModelWorker-->>Scheduler: logits_output, next_token_ids, can_run_cuda_graph
            deactivate TpModelWorker
            deactivate Scheduler

            activate Scheduler
            Scheduler->>Scheduler: process_batch_result
            alt is_prefill
                Scheduler->>SchedulerOutputProcessorMixin: process_batch_result_prefill
                activate SchedulerOutputProcessorMixin
                SchedulerOutputProcessorMixin->>SchedulerOutputProcessorMixin:stream_output
                SchedulerOutputProcessorMixin->>SchedulerOutputProcessorMixin: stream_output_generation
                SchedulerOutputProcessorMixin->>DetokenizerManager: send_to_detokenizer 
                deactivate SchedulerOutputProcessorMixin
            else is_decocer
                Scheduler->>SchedulerOutputProcessorMixin: process_batch_result_decode
                activate SchedulerOutputProcessorMixin
                SchedulerOutputProcessorMixin->>SchedulerOutputProcessorMixin:stream_output
                SchedulerOutputProcessorMixin->>SchedulerOutputProcessorMixin: stream_output_generation
                SchedulerOutputProcessorMixin->>DetokenizerManager: send_to_detokenizer 
                deactivate SchedulerOutputProcessorMixin
            end
            Scheduler->>TokenizerManager: send_to_tokenizer
            deactivate Scheduler
        end
        loop event_loop
            activate DetokenizerManager
            Scheduler->>DetokenizerManager: recv_from_scheduler
            DetokenizerManager->>DetokenizerManager: _request_dispatcher
            DetokenizerManager->>TokenizerManager: send_to_tokenizer
            deactivate DetokenizerManager
        end
        TokenizerManager->>TokenizerManager: handle_loop
        TokenizerManager->>TokenizerManager: _handle_batch_output
        TokenizerManager->>TokenizerManager: _wait_one_response
        TokenizerManager->>Engine:yield out

