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

    model_access = "/home/zack/Downloads/m3e-base"

    span ="I am a span. A short span, but nonetheless a span"

    model_raw = SentenceTransformer(model_access, device="cuda")

    model_pipeline = transformers.FeatureExtractionPipeline(
        model=transformers.AutoModel.from_pretrained(model_access),
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_access, use_fast=False),
        framework="pt",
        device=-1
    )

    config = model_pipeline.model.config
    tokenizer = model_pipeline.tokenizer

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = convert_graph_to_onnx.infer_shapes(
            model_pipeline, 
            "pt"
        )
        ordered_input_names, model_args = convert_graph_to_onnx.ensure_valid_input(
            model_pipeline.model, tokens, input_names
        )

    print("dynamic_axes",dynamic_axes)
    del dynamic_axes["output_0"] # Delete unused output
    del dynamic_axes["output_1"] # Delete unused output
    print("dynamic_axes",dynamic_axes)
    output_names = ["sentence_embedding"]
    dynamic_axes["sentence_embedding"] = {0: 'batch'}


    class SentenceTransformer1(transformers.BertModel):
        def __init__(self, config):
            super().__init__(config)
            # Naming alias for ONNX output specification
            # Makes it easier to identify the layer
            self.sentence_embedding = torch.nn.Identity()

        def forward(self, input_ids,token_type_ids, attention_mask):
            # Get the token embeddings from the base model
            token_embeddings = super().forward(
                input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return self.sentence_embedding(sum_embeddings / sum_mask)

    # Create the new model based on the config of the original pipeline
    model = SentenceTransformer1(config=config).from_pretrained(model_access)

    assert np.allclose(
        model_raw.encode(span),
        model(**tokenizer(span, return_tensors="pt")).squeeze().detach().numpy(),
        atol=1e-6,
    )

    # print("model_raw",model_raw.encode(span))
    # print(model(**tokenizer(span, return_tensors="pt")))

    print("model_args",model_args)
    print("input_names",input_names)

    output="faq_embeding.onnx"

    torch.onnx.export(
            model,
            model_args,
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

    model_input = tokenizer.encode_plus(span)
    print("model_input",model_input)
    model_input = {name : np.atleast_2d(value) for name, value in model_input.items()}
    onnx_result = sess.run(None, model_input)

    assert np.allclose(model_raw.encode(span), onnx_result, atol=1e-5)
    assert np.allclose(
        model(**tokenizer(span, return_tensors="pt")).squeeze().detach().numpy(), 
        onnx_result, 
        atol=1e-5
    )

    print(tokenizer(span, return_tensors="pt"))
    tokens = tokenizer.tokenize(span)
    print("################",tokens)



