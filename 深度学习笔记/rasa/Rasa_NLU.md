# Rasa NLU  
0.14.6
## rasa nlu安装
	pip安装
	pip install rasa_nlu
	源码安装
	git clone https://github.com/RasaHQ/rasa_nlu.git
	cd rasa_nlu
	pip install -r requirements.txt
	pip install -e .

## nlu.md  
NLU训练数据

    ## intent:greet  
    - hey  
    - hello  
    - hi  
    - good morning  
    - good evening  
    - hey there  
  
## nlu_config.yml  
NLU pipeline配置    
  
    language: en  语言配置，中文为zh  
    pipeline: tensorflow_embedding    
  
## 训练NLU model  
	python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose    
### 参数说明
-o PATH, --path PATH：训练后模型存放目录    
-d DATA, --data DATA：训练数据文件或者训练数据文件夹；训练后的模型存放在models/current/nlu目录下  
-u URL, --url URL：训练数据服务器地址  
--endpoints ENDPOINTS：endpoints.yml中定义训练数据服务器地址  
-c CONFIG, --config CONFIG：配置文件，其中定义nlu pipeline和语言  
-t NUM_THREADS, --num_threads NUM_THREADS：训练线程数  
--project PROJECT：model所早的工程名  
--fixed_model_name FIXED_MODEL_NAME：如果存在，模型将始终保留在指定的目录中，而不是创建像'model_20171020-160213'这样的文件夹  
--storage STORAGE：设置存储模型的远程位置。 例如在AWS上。 如果未配置任何内容，则将模型保存在path设置的目录下  
--debug：设置log等级为debug  
-v, --verbose：设置log等级为info      

### 使用python进行训练
	from rasa_nlu.training_data import load_data
	from rasa_nlu.model import Trainer
	from rasa_nlu import config
	
	training_data = load_data(resource_name='nlu_data/',language='zh')
	trainer = Trainer(config.load("config/nlu_config.yml"))
	trainer.train(training_data)
	model_directory = trainer.persist(path='models',project_name="weather",fixed_model_name="nlu")  # 返回nlu模型的储存位置

## 评估NLU model  
	使用测试集评估    
	python -m rasa_nlu.evaluate --data data/examples/rasa/demo-rasa.json --model projects/default/model_20180323-145833    
	或者交叉验证评估    
	python -m rasa_nlu.evaluate --data data/examples/rasa/demo-rasa.json --config sample_configs/config_spacy.yml \    
	--mode crossvalidation    
  
## 运行NLU model
	# 在rasa_core中运行nlu  
	python -m rasa_core.run -d models/dialogue -u models/current/nlu  
	# 独立运行rasa_nlu server  
	python -m rasa_nlu.server --path /path/to/models/ 运行model下所有项目  
	python -m rasa_nlu.server -c config.yml --path /path/to/models/ 使用config.yml配置运行models下所有项目  
	python -m rasa_nlu.server -c config.yml --pre_load hotels --path /path/to/models/  使用config.yml配置运行models下hotels项目     

### 参数说明  
* -P PORT, --port PORT：指定服务端口  
* --pre_load PRE_LOAD [PRE_LOAD ...]：在启动服务器之前将模型预加载到内存中。 如果将“all”作为输入，则将加载所有模型。 否则，您可以指定特定项目名称的列表。 例如：python -m rasa_nlu.server --pre_load project1 --path projects -c config.yaml  
* -t TOKEN, --token TOKEN：身份验证令牌。 如果设置，则拒绝不提供此标记作为查询参数的请求  
* -w WRITE, --write WRITE：log文件  
* --path PATH：模型文件路径  
* --cors [CORS [CORS ...]]：允许CORS（跨域资源共享）调用的域模式的问题。 默认值为`[]`，禁止所有CORS请求。  
* --max_training_processes MAX_TRAINING_PROCESSES：训练进程数。 增加此值将对内存使用产生很大影响。 建议保留默认值。    
* --num_threads NUM_THREADS：用于处理解析请求的并行线程数  
* --endpoints ENDPOINTS：endpoints.yml,配置其他服务器地址  
* --wait_time_between_pulls WAIT_TIME_BETWEEN_PULLS：NLU模型服务器查询之间的等待时间（秒）  
* --response_log RESPONSE_LOG：保存日志的目录（包含查询和响应）。如果设置为“null”，则将禁用日志记录。  
* --storage STORAGE：设置存储模型的远程位置。 例如。 在AWS上。 如果未配置任何内容，则存储在path设置的路径  
* -c CONFIG, --config CONFIG：用于训练模型的配置文件  
* --debug：日志等级设置为DEBUG  
* -v, --verbose：日志等级设置为INFO  
    
## 测试NLU model
### 使用http接口测试
	curl -XPOST localhost:5000/parse -d '{"q":"今天天气怎么样？", "project": "rasa_nlu_test", "model": "model_20170921-170911"}' | python -mjson.tool  
	q：必选项，请求的text  
	project：可选项，如果没有提供，会加载‘default’ project  
	model:可选项，如果没有提供，会加载最新训练的model  

### 使用python脚本进行测试
	from rasa_nlu.model import Interpreter
	interpreter = Interpreter.load("models/weather/nlu/")
	intent = interpreter.parse(u"今天深圳天气怎么样")
	print(intent)

## HTTP接口
### POST /parse
	curl -XPOST localhost:5000/parse -d '{"q":"今天天气怎么样？", "project": "rasa_nlu_test", "model": "model_20170921-170911"}' | python -mjson.tool  
	q：必选项，请求的text  
	project：可选项，如果没有提供，会加载‘default’ project  
	model:可选项，如果没有提供，会加载最新训练的model  

### POST /train
可以将训练数据发布到endpoint训练项目的新模型，请求将等待服务器答复：模型已成功训练或训练退出并出现错误  
使用HTTP服务器，必须指定要训练新模型的项目，以便稍后在解析请求期间使用它：/train?project = my_project。 模型的配置应作为请求内容的一部分：  
例如：
curl -XPOST -H "Content-Type: application/x-yml" localhost:5000/train?project=myproject&model=my_model_name  --data-binary @data/config_train_server_md.yml  
其中config_train_server_md.yml为以下markdown数据格式   
**不能使用训练好的模型来进行重训练**  

	#json数据格式
	language: "en"

	pipeline: "spacy_sklearn"
	
	# data contains the same json, as described in the training data section
	data: {
	  "rasa_nlu_data": {
	    "common_examples": [
	      {
	        "text": "hey",
	        "intent": "greet",
	        "entities": []
	      }
	    ]
	  }
	}
  
	#Markdown数据格式
	language: "en"
	
	pipeline: "spacy_sklearn"
	
	# data contains the same md, as described in the training data section
	data: |
	  ## intent:affirm
	  - yes
	  - yep
	
	  ## intent:goodbye
	  - bye
	  - goodbye

### POST /evaluate
curl -XPOST localhost:5000/evaluate?project=my_project&model=model_XXXXXX -d @data/examples/rasa/demo-rasa.json | python -mjson.tool  
使用数据评估指定模型，返回sklearn评估指标（准确性，f1得分，精度以及摘要报告）  

### GET /status
返回所有当前可用的项目的状态（训练或准备好）以及它们在内存中加载的模型。 还返回服务器可用于实现/解析请求的可用项目列表。  
	
	curl localhost:5000/status | python -mjson.tool  

### GET /version
返回当前版本的Rasa NLU实例，以及加载模型所需的最低模型版本。  

	curl localhost:5000/version | python -mjson.tool

### GET /config（会报错，源码中无此接口，待确认）
返回Rasa NLU实例的默认模型配置。  

	curl localhost:5000/config | python -mjson.tool  

### DELETE /models
从服务器内存中卸载模型  

	curl -X DELETE localhost:5000/models?project=my_restaurant_search_bot&model=model_XXXXXX  

## python接口
### 训练模型
	from rasa_nlu.training_data import load_data
	from rasa_nlu.model import Trainer
	from rasa_nlu import config
	training_data = load_data(resource_name='ids_data/',language='zh')
	trainer = Trainer(config.load("config/nlu_config.yml"))
	trainer.train(training_data)
	model_directory = trainer.persist('models',project_name="weather",fixed_model_name="nlu")  
### 使用模型进行预测
	from rasa_nlu.model import Interpreter
	interpreter = Interpreter.load("models/weather/nlu/")
	intent = interpreter.parse(u"哈尔滨")
	print(intent)
### 加载多个模型时减少内存使用
如果创建了多个模型，可以使用缓存来共享数据，以减小内存使用，要使用缓存，需要在加载和训练模型时传递ComponentBuilder。  

	from rasa_nlu.training_data import load_data
	from rasa_nlu import config
	from rasa_nlu.components import ComponentBuilder
	from rasa_nlu.model import Trainer
	
	builder = ComponentBuilder(use_cache=True)      # will cache components between pipelines (where possible)
	#训练
	training_data = load_data('data/examples/rasa/demo-rasa.json')
	trainer = Trainer(config.load("sample_configs/config_spacy.yml"), builder)
	trainer.train(training_data)
	model_directory = trainer.persist('./projects/default/')  # Returns the directory the model is stored in  
	#预测
	from rasa_nlu.model import Interpreter
	from rasa_nlu import config
	
	# 为简单起见，我们将加载相同的模型两次，通常您会想要使用不同模型的元数据
	
	interpreter = Interpreter.load(model_directory, builder)     # to use the builder, pass it as an arg when loading the model
	# the clone will share resources with the first model, as long as the same builder is passed!
	interpreter_clone = Interpreter.load(model_directory, builder)

## endpoint
endpoint.yml作用为配置远程服务器地址，在nlu中，主要用来配置远程模型服务器。  

	model:
	    url: <path to your model>
	    token: <authentication token>   # [optional]
	    token_name: <name of the token  # [optional] (default: token)


**pipeline**    
tensorflow_embedding：使用训练数据重新训练词向量，要求训练数据大于1000，支持任何语种    
spacy_sklearn：使用pre-trained词向量，训练数据小于1000时使用    
mitie and mitie_sklearn：未来版本会被弃用   
多意图：intent_classifier_tensorflow_embedding  
    
    language: "en"  
      
    pipeline:  
    - name: "intent_featurizer_count_vectors"  
    - name: "intent_classifier_tensorflow_embedding"  
      intent_tokenization_flag: true  
      intent_split_symbol: "+"  
  
  
Pre-configured Pipelines：    
  
    pipeline: "spacy_sklearn"  
	等价于  
    pipeline:  
    - name: "nlp_spacy"  
    - name: "tokenizer_spacy"  
    - name: "intent_entity_featurizer_regex"  
    - name: "intent_featurizer_spacy"  
    - name: "ner_crf"  
    - name: "ner_synonyms"  
    - name: "intent_classifier_sklearn"    
      
	pipeline: "tensorflow_embedding"  
	等价于  
	pipeline:  
    - name: "tokenizer_whitespace"  
    - name: "ner_crf"  
    - name: "ner_synonyms"  
    - name: "intent_featurizer_count_vectors"  
    - name: "intent_classifier_tensorflow_embedding"  
	  
	pipeline: "keyword"  
	等价于  
	pipeline:  
    - name: "intent_classifier_keyword"  
	  
mitie配置    
  
    pipeline:  
    - name: "nlp_mitie"  
      model: "data/total_word_feature_extractor.dat"  
    - name: "tokenizer_mitie"  
    - name: "ner_mitie"  
    - name: "ner_synonyms"  
    - name: "intent_entity_featurizer_regex"  
    - name: "intent_classifier_mitie"    
  
mitie_sklearn配置：    
      
    pipeline:  
    - name: "nlp_mitie"  
      model: "data/total_word_feature_extractor.dat"  
    - name: "tokenizer_mitie"  
    - name: "ner_mitie"  
    - name: "ner_synonyms"  
    - name: "intent_entity_featurizer_regex"  
    - name: "intent_featurizer_mitie"  
    - name: "intent_classifier_sklearn"    
  
  
## 训练数据格式  
Rasa NLU的训练数据分为不同部分：示例(examples)，同义词(synonyms)，正则表达式功能(regex features)和查找表(lookup tables)。    
  
### Markdown 格式  
使用无序列表语法列出示例，例如，减号 - ，星号*或加+。 示例按意图分组，实体以链接方式表示。    
  
    ## intent:check_balance  <!--意图：以“## ”开头-->  
    - what is my balance <!-- 没有标注实体 -->  
    - how much do I have on my [savings](source_account) <!-- 实体"source_account" 的value为"savings" -->  
    - how much do I have on my [savings account](source_account:savings) <!--同义词，第一种表达方式-->  
    - Could I pay in [yen](currencies)?  <!-- 在查找表currencies中匹配实体 -->  
      
    ## intent:greet  
    - hey  
    - hello  
      
    ## synonym:savings   <!-- 同义词，第二种表示方式 -->  
    - savings account  
      
    ## regex:zipcode  <!-- 正则表达方式 -->  
    - [0-9]{5}  
      
    ## lookup:currencies   <!-- 查找表列表形式，匹配时忽略查找表中单词大小写 -->  
    - Yen  
    - USD  
    - Euro  
      
    ## lookup:additional_currencies  <!-- 通过文件方式导入查找表，txt中以换行符进行断句，匹配时忽略查找表中单词大小写 -->  
    path/to/currencies.txt  
  
### JSON格式  
json训练数据按如下格式书写    
  
    {  
    	"rasa_nlu_data": {  
    		"common_examples": [],  
    		"regex_features" : [],  
    		"lookup_tables"  : [],  
    		"entity_synonyms": []  
    	}  
    }     
  
#### Common Examples    
	{  
	  "text": "上海明天的天气",  <!--text为用户说的话-->    
	  "intent": "weather_address_date-time",  <!--text的意图-->    
	  "entities": [   <!--text中实体列表，实体可以跨越多个单词-->    
	    {  
	      "start": 0,  <!--实体对应text中位置-->  
	      "end": 2,  
	      "value": "上海",  <!--value字段不必与示例中的子字符串完全对应。 这样，您可以将同义词或拼写错误映射到相同的值。-->  
	      "entity": "address"  
	    }  
		{  
	      "start": 2,  
	      "end": 4,  
	      "value": "明天",  
	      "entity": "date-time"  
	    }  
	  ]  
	}  
  
#### 同义词（需要在pipline中添加ner_synonyms）  
1. 在common example中将两个词定义为同一个value，那么两个词会被视为同义词    
例如：    

		[  
		  {  
		    "text": "in the center of NYC",  
		    "intent": "search",  
		    "entities": [  
		      {  
		        "start": 17,  
		        "end": 20,  
		        "value": "New York City",  
		        "entity": "city"  
		      }  
		    ]  
		  },  
		  {  
		    "text": "in the centre of New York City",  
		    "intent": "search",  
		    "entities": [  
		      {  
		        "start": 17,  
		        "end": 30,  
		        "value": "New York City",  
		        "entity": "city"  
		      }  
		    ]  
		  }  
		]	  
2. 定义同义词数组    
例如：    

		{  
		  "rasa_nlu_data": {  
		    "entity_synonyms": [  
		      {  
		        "value": "New York City",  
		        "synonyms": ["NYC", "nyc", "the big apple"]  
		      }  
		    ]  
		  }  
		}  
  
#### 正则表达式功能（当前版本只有ner_crf支持）  
正则表达式可用于支持意图分类和实体提取。 例如。 如果您的实体具有邮政编码中的某种结构，则可以使用正则表达式来轻松检测该实体。 对于zipcode示例，如下所示：    
  
	{  
	    "rasa_nlu_data": {  
	        "regex_features": [  
	            {  
	                "name": "zipcode",  
	                "pattern": "[0-9]{5}"  
	            },  
	            {  
	                "name": "greet",  
	                "pattern": "hey[^\\s]*"  
	            },  
	        ]  
	    }  
	}    
#### 查找表（大小写不敏感）    
1. 外部文件提供查找表，各个元素以换行符进行分隔    
例如：   
 
		{  
		    "rasa_nlu_data": {  
		        "lookup_tables": [  
		            {  
		                "name": "plates",  
		                "elements": "data/test/lookup_tables/plates.txt"  
		            }  
		        ]  
		    }  
		}  
2. 以列表的形式提供查找表    
例如：    

		{  
		    "rasa_nlu_data": {  
		        "lookup_tables": [  
		            {  
		                "name": "plates",  
		                "elements": ["beans", "rice", "tacos", "cheese"]  
		            }  
		        ]  
		    }  
		}						  
要使查找表有效，训练数据中必须有一些匹配示例。 否则模型将不会学习使用查找表匹配功能。  

## NER-命名实体识别

| 组件 | 使用的库 | 模型 | 说明 |
| ---- | ------- | ---- | ---- |
| ner_crf | sklearn-crfsuite | 条件随机场（crf） | 适用于定制实体 |
| ner_spacy | spaCy | 均值感知 | 提供pre-trained entities |
| ner_duckling_http/ner_duckling | duckling | 上下文无关文法 | 提供pre-trained entities |
| ner_mitie | MITIE | 结构化SVM | 适用于定制实体 |
**MITIE**  
MITIE是MIT的NLP 团队发布的一个信息抽取库和工具，包含：命名实体抽取（Named-Entity-Recognize，NER）和二元关系检测功能（Bianry relation detection）  
另外也提供了训练自定义抽取器和关系检测器的工具。它的主要工作在于：  
1.distributional word embeddings：简单说来就是将每个词映射到向量空间，向量之间的距离代表词之间的相似度，且在语义、语法两种意义上。关于词向量最新的研究可以看Learned in translation: contextualized word vectors；
2.Structural Support Vector Machines  
  
Duckling 是 Facebook 出品的一款用 Haskell 语言写成的 NER 库，基于规则和模型。Duckling 支持多种实体的提取  
spacy:sapcy是python语言处理库，可以完成词性分析、命名实体识别、依赖关系刻画、词向量近似度计算、词语降维和可视化    

## 返回数据
Rasa NLU返回用户说话的intent和置信度，每个intent含有一个实体列表以及对应实体的执行度；  
可以使用置信度分数选择何时忽略Rasa NLU的预测和触发回退行为，例如要求用户重新措辞。 如果使用的是Rasa Core，则可以使用Fallback Policy.执行此操作。   


## 组件(Component)说明

| 组件名称 | 说明 | 描述 | 参数 | 备注 |
| ------- | ---- | ---- | ---- | ---- |
| nlp_mitie | mitie初始化 | 初始化mitie组件，每个mitie组件都依赖于此，因此应将其放在使用任何mitie组件的pipeline的开头 | mitie训练不同语言数据后的模型，model: "data/total_word_feature_extractor.dat" | - | 
| nlp_spacy | spacy语言初始化 | 初始化spacy结构。 每个spacy组件都依赖于此，因此应将其放在使用任何spacy组件的每个pipeline的开头 | model: "en_core_web_md" 模型文件；<br> case_sensitive: false 英文是否大小写敏感 | - |
| intent_featurizer_mitie | MITIE intent特征化 | 使用MITIE特征将intent特征化，提供给MITIE intent分类器 | - | 目前只能提供给intent_classifier_sklearn组件使用，不能给intent_classifier_mitie 组件使用 |  
| intent_featurizer_spacy | spacy intent特征化 | 使用spacy特征将intent特征化，提供给spacy intent分类器，用作需要意图特征的意图分类器的输入，如：intent_classifier_sklearn | - | - |     
| intent_featurizer_ngrams | 将char-ngram特征附加到特征向量 | 将ngram特征向量附加到特征向量中。 在训练期间，该组件寻找最常见的字符序列（例如app或ing）。 如果字符序列存在于单词序列中，则添加的特征表示布尔标志。 | max_number_of_ngrams: 10 | 在pipeline之前需要有另一个意图功能强化器 |
| intent_featurizer_count_vectors | 创建intent特征的词袋表示 | 创建intent特征的词袋表示,用作意图分类器的输入，该分类器需要意图特征的词袋表示（例如，intent_classifier_tensorflow_embedding） | "analyzer": 'word',  # use 'char' or 'char_wb' for character <br>"token_pattern": r'(?u)\b\w\w+\b' <br>"strip_accents": None  # {'ascii', 'unicode', None} <br>"stop_words": None  # string {'english'}, list, or None (default) <br>"min_df": 1  # float in range [0.0, 1.0] or int <br>"max_df": 1.0  # float in range [0.0, 1.0] or int <br>"min_ngram": 1  # int <br>"max_ngram": 1  # int <br>"max_features": None  # int or None <br>"lowercase": true  # bool <br>"OOV_token": None  # string or None <br>"OOV_words": []  # list of strings | - |
| intent_classifier_keyword | 简单的关键字匹配意图分类器 | 此分类器主要用作占位符。 通过在传递的消息中搜索这些关键字，它能够识别hello和goodbye意图。 | - | - |
| intent_classifier_mitie | MITIE意图分类器（使用文本分类） | 此分类器使用MITIE执行意图分类。 底层分类器使用具有稀疏线性内核的多类线性SVM。 | - | - |
| intent_classifier_sklearn | sklearn意图分类器 | sklearn意图分类器训练线性SVM，使用网格搜索对其进行优化。 还提供标签的排名 | C: [1, 2, 5, 10, 20, 100]，指定要对C-SVM进行交叉验证的正则化值列表。与GridSearchCV中的kernel超参数一起使用。 <br>kernels: ["linear"],指定与C-SVM一起使用的内核。这与GridSearchCV中的C超参数一起使用。 | - |
| intent_classifier_tensorflow_embedding | 嵌入意图分类器 | 嵌入意图分类器将用户输入和意图标签嵌入到同一空间中。 监督嵌入通过最大化它们之间的相似性来训练。嵌入意图分类器需要在pipeline中以特征化组件开头。 此特征创建用于嵌入的功能。 建议使用intent_featurizer_count_vectors，可以选择在其前面加上nlp_spacy和tokenizer_spacy。 | 待补充 | - |
| intent_entity_featurizer_regex | 正则表达式功能创建，以支持意图和实体分类 | 训练数据中存在正则表达式，需要使用该组件 | - |  用于实体提取的正则表达式功能目前仅由ner_crf组件支持 <br>pipeline中在此组件之前需要存在一个tokenizer |
| tokenizer_whitespace | 使用空格作为分隔符的标记生成器 | 为每个空格分隔的字符序列创建一个标记。 可用于为MITIE实体提取器定义标记 | - | - |
| tokenizer_jieba | 使用jieba进行中文分词 | 使用jieba标记器专门为中文创建标记，可用于为MITIE实体提取器定义标记 | dictionary_path: "path/to/custom/dictionary/dir"，用户自定义词典 | - |
| tokenizer_mitie | 使用MITIE的标记生成器 | 使用MITIE标记生成器创建标记。 可用于为MITIE实体提取器定义标记。 | - | - |
| tokenizer_spacy | 使用spacy的标记生成器 | 使用spacy标记生成器创建标记。 可用于为MITIE实体提取器定义标记 | - | - |
| ner_mitie | MITIE实体提取 | 使用MITIE提取消息中的实体。 底层分类器使用具有稀疏线性内核和自定义功能的多类线性SVM。 MITIE组件不提供实体置信度值。 | - | - |
| ner_spacy | spacy实体提取 | 使用spacy预测消息的实体。 spacy使用statistical BILOU transition模型。 截至目前，该组件只能使用spacy内置实体提取模型而无法重新训练。 此提取器不提供任何置信度分数。 | - | - |
| ner_synonyms | 将同义实体值映射到相同的值。 | 如果训练数据包含已定义的同义词,此组件将确保检测到的实体值将映射到相同的值 | - | - |
| ner_crf | 条件随机场实体提取 | CRF可以被认为是无向马尔可夫链，其中时间步长是单词，状态是实体类。 单词的特征（大写，POS标记等）给出了某些实体类的概率，以及相邻实体标签之间的转换：然后计算并返回最可能的标签集。 如果使用POS功能（pos或pos2），则必须安装spaCy | features: [["low", "title"], ["bias", "suffix3"], ["upper", "pos", "pos2"]] <br>BILOU_flag: true 仅在每个实体超过100个示例时使用 <br>max_iterations: 50 <br>L1_c: 0.1 L1正则化系数 <br>L2_c: 0.1 L2正则化系数 | - | - |
| ner_duckling_http | Duckling允许您以多种语言(不支持中文，呵呵)提取日期，金额，距离等常见实体 | 要使用此组件，需要运行一个Duckling服务器。 最简单的选择是使用docker run -p 8000：8000 rasa / duckling来启动docker容器 | url: "http://localhost:8000" <br>dimensions: ["time", "number", "amount-of-money", "distance"] <br>locale: "de_DE" <br>timezone: "Europe/Berlin" | - |

## 客制化组件
实现请参考github：[https://github.com/user-ZJ/rasa_nlu_zh](https://github.com/user-ZJ/rasa_nlu_zh)  
可以创建自定义组件来执行NLU当前不提供的特定任务（例如，情绪分析）  
下面以添加pkuseg(0.0.21)分词组件来替换jieba分词为例  
1. 在rasa_nlu/rasa_nlu/tokenizers目录下创建pkuseg_tokenizer.py文件，并创建PkusegTokenizer类。  
2. 定义类中成员
	
		# 在pipeline中使用的名称
	    name = "tokenizer_pkuseg"
	    # 组件在调用时提供的属性
	    provides = ["tokens"]
	    # 语言支持列表
	    language_list = ["zh"]
	    # 此组件需要pipeline中前一个组件提供的属性，如果require包含“tokens”，则管道中的前一个组件需要在上述“provide”属性中具有“tokens”
	    requires = []
	    #定义组件的默认配置参数，这些值可以在模型的管道配置中覆盖。 组件应选择合理的默认值，并且应该能够使用默认值创建合理的结果。
	    defaults = {
	        "dictionary_path": None  # default don't load custom dictionary
	    }
3. 实现组件中必须的函数  

		train():训练组件
		process():处理传入的消息。
		load():从文件加载此组件。
		required_package():指定需要安装哪些python包才能使用此组件
		persist():将此组件保存到磁盘以供将来加载  
4. 在注册pkuseg组件
在rasa_nlu/registry.py文件中注册pkuseg组件  

		
	 
 





## F&Q:
1. "error": "No project found with name 'weather'."  
> 使用python -m rasa_nlu.server --path models/weather/nlu/启动nlu server，使用curl -XPOST localhost:5000/parse -d '{"q":"hello there", "project": "weather", "model": "nlu"}获取意图，会报"error": "No project found with name 'weather'."     
> **解决方案：**   
> 使用python -m rasa_nlu.server --path models启动nlu server  

2. jieba分词不准确
> a. 词不能被分开，如“今天天气”不能被分为('今天','天气')  
> b. 不该分开的词被分开，如“下个星期五”被分为('下个星期','五')  
> **解决方案：**  
> **情景a：**  
> 1.使用suggest_freq(segment, tune=True) 调节单个词语的词频，使其能（或不能）被分出来    
> 
>     tokenized = jieba.tokenize('今天天气')  
>     for (word, start, end) in tokenized:  
>        print(word, start, end)  
>     print(jieba.get_FREQ('今天天气'))  
>     print(jieba.suggest_freq(('今天', '天气'), True)) #使'今天天气'被分为两个词    
>     print(jieba.get_FREQ('今天天气'))  
>     tokenized = jieba.tokenize('今天天气')    
>     for (word, start, end) in tokenized:  
>         print(word, start, end)    
> 2.使用自定义词典,并将词频设置为0  
> 创建自定义词典dict.txt，并添加"今天天气 0"  
> 
>     jieba.load_userdict("dict.txt")   
>     tokenized = jieba.tokenize('今天天气')  
>     for (word, start, end) in tokenized:   
>         print(word, start, end)   
>     
> **情景b：**  
> 同情景a，  
> 1.jieba.suggest_freq('下个星期五', True)  
> 2.在词典中添加'下个星期五' 

