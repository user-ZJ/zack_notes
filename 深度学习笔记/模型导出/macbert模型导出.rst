macbert模型导出
=================================

.. code-block:: python

    import time
    import pprint
    import multiprocessing
    from pathlib import Path
    import onnx
    import torch
    import transformers
    import os
    import numpy as np
    import onnxruntime as ort
    from transformers import convert_graph_to_onnx,AutoTokenizer
    from transformers import BertTokenizer, BertForMaskedLM
    import operator
    from transformers.convert_graph_to_onnx import convert

    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details


    model_access = "data/macbert4csc-base-chinese"
    onnx_path = "onnx/macbert4csc.onnx"

    texts = ["今天新情很好", "你找到你最喜欢的工作，我也很高心。"]

    tokenizer = BertTokenizer.from_pretrained(model_access)
    model = BertForMaskedLM.from_pretrained(model_access)
    config = model.config
    # print(model)
    # print(config)

    tokenized_tokens = tokenizer(texts, padding=True, return_tensors='pt')
    print(tokenized_tokens)
    outputs = model(input_ids=tokenized_tokens['input_ids'],token_type_ids=tokenized_tokens['token_type_ids'],attention_mask=tokenized_tokens['attention_mask'])
    # outputs = model(**tokenizer(texts, padding=True, return_tensors='pt'))
    print(outputs.logits.size())
    logits = torch.argmax(outputs.logits, dim=-1)
    print(torch.argmax(outputs.logits, dim=-1))
    for l in logits:
        print(tokenizer.decode(l))

    class macbert4cscExport(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self,input_ids,token_type_ids,attention_mask):
            # Get the token embeddings from the base model
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            logits = torch.argmax(outputs.logits, dim=-1)
            return logits

    # # Create the new model based on the config of the original pipeline
    tokenized_tokens = tokenizer(texts, padding=True, return_tensors='pt')
    print(tokenized_tokens)
    macbert4csc_model = macbert4cscExport(model)
    export_output = macbert4csc_model.forward(tokenized_tokens["input_ids"],tokenized_tokens['token_type_ids'],tokenized_tokens['attention_mask'])
    print(export_output)

    assert np.allclose(logits.numpy(),export_output,atol=1e-6)

    export_text = texts[0]*1000
    export_text = export_text[:510]
    export_tokens = tokenizer(export_text, padding=True, return_tensors='pt')
    print(export_tokens)
    export_inputs = (export_tokens['input_ids'],export_tokens['token_type_ids'],export_tokens['attention_mask'])



    torch.onnx.export(
            macbert4csc_model,
            export_inputs,
            f=onnx_path,
            input_names=['input_ids','token_type_ids','attention_mask'],
            output_names=['logits'],
            dynamic_axes={"input_ids":{0:'batch_size',1:'seq'},"token_type_ids":{0:'batch_size',1:'seq'},"attention_mask":{0:'batch_size',1:'seq'},"logits":{0:'batch_size',1:'seq'}},
            do_constant_folding=True,
            export_params=True,
            opset_version=12,
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')



    input_ids = tokenized_tokens['input_ids'].numpy().astype(np.int64)
    token_type_ids = tokenized_tokens['token_type_ids'].numpy().astype(np.int64) 
    attention_mask = tokenized_tokens['attention_mask'].numpy().astype(np.int64)

    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # opt.log_severity_level = 3
    # opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(str(onnx_path), opt) # Loads the model

    model_input = {'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask':attention_mask}
    # print("model_input",model_input)
    onnx_result = sess.run(None, model_input)

    assert np.allclose(logits.numpy(), onnx_result, atol=1e-5)
    # assert np.allclose(
    #     model(**tokenizer(span, return_tensors="pt")).squeeze().detach().numpy(), 
    #     onnx_result, 
    #     atol=1e-5
    # )




