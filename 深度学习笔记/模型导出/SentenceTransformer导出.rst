SentenceTransformer导出
==============================================

.. code-block:: python

    import time
    import pprint
    import multiprocessing
    from pathlib import Path

    import onnx
    import torch
    import transformers

    import numpy as np
    import onnxruntime as rt

    from termcolor import colored
    from sentence_transformers import SentenceTransformer,models
    from transformers import convert_graph_to_onnx,AutoTokenizer

    pp = pprint.PrettyPrinter(indent=4)
    pprint = pp.pprint

    model_access = "xiaobu-embedding-v2"

    span ="I am a span. A short span, but nonetheless a span"
    span_list = [span]

    model_raw = SentenceTransformer(model_access, device="cpu")

    print(len(model_raw.encode(span)))

    tokens = model_raw.tokenize(span_list)
    print(tokens)
    output = model_raw.forward(model_raw.tokenize(span_list))
    print(output["sentence_embedding"].size())

    assert np.allclose(
        model_raw.encode(span),
        output["sentence_embedding"].squeeze().detach().numpy(),
        atol=1e-6,
    )


    class SentenceTransformer1(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids,token_type_ids, attention_mask):
            # Get the token embeddings from the base model
            outputs = self.model.forward(
                {"input_ids":input_ids, 
                "token_type_ids":token_type_ids,
                "attention_mask":attention_mask, 
                }
            )
            return outputs["sentence_embedding"]

    # Create the new model based on the config of the original pipeline
    model = SentenceTransformer1(model_raw)
    assert np.allclose(
        model.forward(tokens['input_ids'],tokens['token_type_ids'],tokens['attention_mask']).squeeze().detach().numpy(),
        output["sentence_embedding"].squeeze().detach().numpy(),
        atol=1e-6,
    )

    input_names = ['input_ids',  'token_type_ids','attention_mask']
    output_names = ["embedding"]
    dynamic_axes = {'input_ids': {0: 'batch', 1: 'sequence'}, 'token_type_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'}, 'embedding': {0: 'batch', 1: 'sequence'}}
    export_inputs = (tokens['input_ids'],tokens['token_type_ids'],tokens['attention_mask'])

    output="embeding.onnx"

    torch.onnx.export(
            model,
            export_inputs,
            f=output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            opset_version=12,
        )

    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')

    opt = rt.SessionOptions()
    opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    # opt.log_severity_level = 3
    # opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

    sess = rt.InferenceSession(str(output), opt) # Loads the model

    model_input = tokens
    model_input = {name : value.numpy() for name, value in model_input.items()}
    onnx_result = sess.run(None, model_input)

    assert np.allclose(model_raw.encode(span), onnx_result, atol=1e-5)






