structbert模型导出
============================

模型地址：https://www.modelscope.cn/models/iic/nlp_structbert_sentence-similarity_chinese-large/summary

.. code-block:: python 

    from funasr.models.emotion2vec.model import Emotion2vec
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models import Model
    import torch
    import soundfile as sf
    import torch.nn.functional as F
    import numpy as np
    import onnxruntime as rt

    model_path = "/nasdata/0003_model/nlp/nlp_structbert_sentence-similarity_chinese-large"
    onnx_path = "structbert.onnx"

    text1 = '安全帽在合格有效期内，帽撑完好，可以使用'
    text2 = "可以了，可以开始了。检查安全帽，有合格证，在三十个月有效期内，外观良好，无破损。"

    inference_pipeline = pipeline(Tasks.sentence_similarity, '/nasdata/0003_model/nlp/nlp_structbert_sentence-similarity_chinese-large')
    sim2 = inference_pipeline(input=(text1, text2))
    print(sim2)


    model = inference_pipeline.model.cpu()
    tokenizer = inference_pipeline.preprocess
    id2label = inference_pipeline.id2label
    print(id2label)

    tokened = tokenizer((text1, text2))
    print(tokened)
    input_ids=tokened['input_ids']
    token_type_ids=tokened['token_type_ids']
    attention_mask=tokened['attention_mask']
    out = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
    print(out)
    logits = torch.softmax(out["logits"],1).detach()
    print(logits)


    class StructBertExport(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids,token_type_ids,attention_mask):
            out = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
            logits = torch.softmax(out["logits"],1)
            return logits
        
    export_model = StructBertExport(model)

    torch.onnx.export(
            export_model,
            (input_ids,token_type_ids,attention_mask),
            onnx_path,
            input_names=["input_ids","token_type_ids","attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids":{1:"len"},"token_type_ids":{1:"len"},"attention_mask":{1:"len"}},
            do_constant_folding=True,
            export_params=True,
            opset_version=12,
        )
    print(input_ids.size())

    opt = rt.SessionOptions()
    opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = rt.InferenceSession(onnx_path, opt)

    model_input = {"input_ids" : input_ids.numpy(),"token_type_ids":token_type_ids.numpy(),"attention_mask":attention_mask.numpy()}
    onnx_result = sess.run(None, model_input)
    print(onnx_result)
    assert np.allclose(logits.numpy(), onnx_result, atol=1e-5)


    input_ids = [[101, 2128, 1059, 2384, 1762, 1394, 3419, 3300, 3126, 3309, 1079, 8024,
            2384, 3053, 2130, 1962, 8024, 1377,  809,  886, 4500,  102, 1377,  809,
            749, 8024, 1377,  809, 2458, 1993,  749,  511, 3466, 3389, 2128, 1059,
            2384, 8024, 3300, 1394, 3419, 6395, 8024, 1762,  676, 1282,  702, 3299,
            3300, 3126, 3309, 1079, 8024, 1912, 6225, 5679, 1962, 8024, 3187, 4788,
            2938,  511,  102]]
    token_type_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    attention_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    model_input = {"input_ids" : np.array(input_ids),"token_type_ids":np.array(token_type_ids),"attention_mask":np.array(attention_mask)}
    onnx_result = sess.run(None, model_input)
    print(onnx_result)