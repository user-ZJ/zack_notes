# RASA Core  
0.13.3
## stories.md  
	## story1            story以##开头，名称为可选项    
	* greet              intent,以\*开头    
	   - utter_greet    \\bot采取的action，以\-开头,action中可访问api或者与外界互动    
	* inform{"location": "rome", "price": "cheap"}    用户说话, 格式为intent{"entity1": "value", "entity2": "value"}    
	   - action_on_it  
	   - action_ask_cuisine  
	* inform{"cuisine": "spanish"}  
	   - action_ask_numpeople        要采取的动作  
	* inform{"people": "six"}  
	   - action_ack_dosearch     
	* affirm OR thankyou            多个意图使用同一个动作    
  	   - action_handle_affirmation  
  
### Checkpoints    
可以使用>Checkpoints来模块化和简化您的训练数据。检查点可能很有用，但不要过度使用它们。 使用大量检查点可以快速使您的示例故事难以理解。如果在不同的故事中经常重复故事块，则使用检查点是有意义的，但没有检查点的故事更容易阅读和书写。

	## first story  
	* hello  
	   - action_ask_user_question  
	> check_asked_question  
	  
	## user affirms question  
	> check_asked_question  
	* affirm  
	  - action_handle_affirmation  
	  
	## user denies question  
	> check_asked_question  
	* deny  
	  - action_handle_denial  
	  
  
## domain.yml  
  
	intents:  
	 - greet{use_entities: false}    #忽略意图的entity  
	 - default  
	 - goodbye  
	 - affirm  
	 - thank_you  
	 - change_bank_details  
	 - simple  
	 - hello  
	 - why  
	 - next_intent  
	  
	entities:  
	 - name  
	  
	slots:  
	  name:  
	    type: text  
	  
	templates:  
	  utter_greet:  
	    - "hey there {name}!"  # {name} will be filled by slot (same name) or by custom action code  
	    - text: "Hey! How are you?"  
	      buttons:  
    	  - title: "great"  
      		payload: "great"  
    	  - title: "super sad"  
      		payload: "super sad"  
	  utter_goodbye:  
	    - "goodbye"  
	      image: "https://i.imgur.com/nGF1K8f.jpg"  
	    - "bye bye"   # multiple templates - bot will randomly pick one of them  
	  utter_default:  
	    - "default message"  
	  
	actions:  
	  - utter_default  
	  - utter_greet  
	  - utter_goodbye  
  
intents：用户说话意图列表，和Rasa NLU中定义一致    
actions：bot可以采取的所有动作    
templates：bot可以说的话的模板字符串,可以包含图片、button、变量      
entities：从用户说的话中提取的信息，和Rasa NLU中定义一致    
slots：槽，在对话期间跟踪的信息（例如用户年龄）    
  
## policy.yml  
配置和策略文件，默认可以写入：    
  
	policies:  
	  - name: KerasPolicy  
	    epochs: 100  
	    max_history: 5  
	  - name: FallbackPolicy  
	    fallback_action_name: 'action_default_fallback'  
	  - name: MemoizationPolicy  
	    max_history: 5  
	  - name: FormPolicy  
  
## endpoints.yml  
端点配置文件，可以配置用户自定义action服务地址，nlg服务器地址    
  
	action_endpoint:  
	  url: http://localhost:5055/webhook  
	nlg:  
	  url: http://localhost:5055/nlg  
	  token: <token>  # [optional]  
      token_name: <name of the token> # [optional] (default: token)  
	core_endpoint:  
	  url: http://localhost:5005  

## credentials.yml
身份验证凭据文件  

    facebook:  
      verify: "rasa-bot"  
      secret: "3e34709d01ea89032asdebfe5a74518"  
      page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD" 
	ocketio:
	  user_message_evt: user_uttered
	  bot_message_evt: bot_uttered

## rasa core安装
	pip安装
	pip install -U rasa_core
	源码安装
	git clone https://github.com/RasaHQ/rasa_core.git
	cd rasa_core
	pip install -r requirements.txt
	pip install -e .
	
## 训练Dialogue Model  
	python -m rasa_core.train -d domain.yml -s stories.md -o models/dialogue -c policies.yml   
训练后的模型存储在models/dialogue目录下    
使用python脚本训练：    
  
	from rasa_core.agent import Agent  
	agent = Agent()  
	data = agent.load_data("stories.md")  
	agent.train(data)  
  
### 训练参数    
	-h, --help：显示帮助信息    
	--augmentation AUGMENTATION：在train期间使用多少数据扩充    
	--dump_stories:如果启用，将展平的故事保存到文件中    
	--debug_plots:如果启用，将创建显示检查点及其在文件中的故事块之间的连接的图    
	-c CONFIG, --config CONFIG：policy的yaml文件    
	-o OUT, --out OUT：训练后模型保存目录    
	-s STORIES, --stories STORIES：story文件或文件夹    
	--url URL：如果提供，则从URL下载story进行训练，通过URL发送get请求来获取数据    
	-d DOMAIN, --domain DOMAIN： domain.yml文件    
	-v, --verbose：设置log等级为INFO      
	-vv, --debug：设置log等级为DEBUG    
	--quiet：设置log等级为WARNING    
  
## 评估模型    
	python -m rasa_core.evaluate --core models/dialogue --stories test_stories.md -o results    
失败的story会被存储到results/failed_stories.md    
评估结果会被存储在results/story_confmat.pdf文件    
  
### 评估参数  
	-h, --help：显示帮助信息     
	-u NLU, --nlu NLU：NLU模型    
	-o OUTPUT, --output OUTPUT：评估创建的文件存储路径    
	--e2e, --end-to-end：对组合动作和意图预测运行端到端评估。 需要端到端格式的story文件。     
	--endpoints ENDPOINTS：连接器的配置文件，endpoint.yml文件      
	--fail_on_prediction_errors:如果遇到预测错误，则抛出异常。    
	--core CORE:core model文件夹    
	-s STORIES, --stories STORIES：story文件或文件夹路径    
	--url URL：如果提供，则从URL下载story进行训练，通过URL发送get请求来获取数据     
	-v, --verbose：设置log等级为INFO      
	-vv, --debug：设置log等级为DEBUG    
	--quiet：设置log等级为WARNING   
  
### 通过HTTP评估模型    
	curl --data-binary @eval_stories.md "localhost:5005/evaluate" | python -m json.tool    
	#端到端评估（e2e）    
	curl --data-binary @eval_stories.md "localhost:5005/evaluate?e2e=true" | python -m json.tool    
  
## 运行Rasa Core  
	python -m rasa_core.run -d models/dialogue    
在没有NLU情况下，单独输入domain中定义的intent可以对Core进行调试，如输入：/greet      
### 运行参数    
	--enable_api：除输入通道外，还启动Web服务器api    
	-d：Rasa Core model路径    
	-u：Rasa NLU model路径    
	-o：log文件路径    
	--endpoints：endpoints.yml，存放自定义action服务地址、nlg服务地址   
	-v, --verbose：设置日志等级为INFO  
	-vv, --debug：设置日志等级为DEBUG  
	--quiet：设置日志等级为WARNING  
	-p PORT, --port PORT:服务启动端口，默认为5005  
	--auth_token AUTH_TOKEN：启用基于令牌的身份验，请求需要提供要接受的令牌  
	--cors [CORS [CORS ...]]：enable CORS for the passed origin. Use * to whitelist all origins  
	--credentials CREDENTIALS：使用yml文件作为连接的身份验证凭据  
	-c CONNECTOR, --connector CONNECTOR：要连接的服务  
	--jwt_secret JWT_SECRET：非对称JWT方法的公钥或对称方法的共享密钥。 请确保使用--jwt_method选择签名方法，否则将忽略此参数。  
	--jwt_method JWT_METHOD：用于签名JWT身份验证有效负载的方法。  
  
## 添加网站凭证    
例如在facebook上使用聊天机器人    
创建credentials.yml文件，写入facebook凭证    
  
    facebook:  
      verify: "rasa-bot"  
      secret: "3e34709d01ea89032asdebfe5a74518"  
      page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"    
  
运行Rasa Core：    
  
	python -m rasa_core.run -d models/dialogue -u models/nlu/current --port 5002 --credentials credentials.yml  
  
## HTTP API  
在查询中返回多个事件：    
  
	curl -XPOST http://localhost:5005/conversations/default/tracker/events -d \  
	    '{"event": "slot", "name": "cuisine", "value": "mexican"}'    
	curl -XPOST localhost:5005/conversations/default/respond -d '{"q":"hello"}'
### 安全认证    
为了使rasa core服务不是暴露给所有人，可以添加http认证    
1. 基于令牌的身份验证（Token Based Auth）     
		启动服务时添加--auth_token thisismysecret参数  
		python -m rasa_core.run \  
		    --enable_api \  
		    --auth_token thisismysecret \  
		    -d models/dialogue \  
		    -u models/nlu/current \  
		    -o out.log    
		访问时需要添加相应的token    
		curl -XGET localhost:5005/conversations/default/tracker?token=thisismysecret    
2. 基于JWT认证    
使用--jwt_secret thisismysecret启用基于JWT的身份验证。 对服务器的请求需要在使用此密钥和HS256算法签名的Authorization头中包含有效的JWT令牌。    
用户必须具有用户名和角色属性。 如果角色是admin，则可以访问所有endpoint。 如果角色是用户，则只有sender_id与用户的用户名匹配时才能访问具有sender_id参数的endpoint。    
		python -m rasa_core.run \  
		    --enable_api \  
		    --jwt_secret thisismysecret \  
		    -d models/dialogue \  
		    -u models/nlu/current \  
		    -o out.log    
请求应该设置正确的JWT标头：    
		"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"  
		                 "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi"  
		                 "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO"  
		                 "Gl8eZFVfKXA6jhncgRn-I"  
  
### 从服务器获取rasa core模型    
可以配置http服务器以从其他URL获取模型：    
  
	python -m rasa_core.run \  
	    --enable_api \  
	    -d models/dialogue \  
	    -u models/nlu/current \  
	    --endpoints my_endpoints.yaml \  
	    -o out.log    
模型服务器在endpoint配置（my_endpoints.yaml）中指定，可以在其中指定服务器URL Rasa Core定期检查压缩的Rasa Core模型：      
  
	models:  
	  url: http://my-server.com/models/default_core@latest  
	  wait_time_between_pulls:  10   # [optional](default: 100)    
	  如果只想从服务器获取一次模型，设置wait_time_between_pulls为None，只有当服务器上zip形式的模型的hash值改变，core才会下载。      
  
## Agent    
Agent是一个简单的api，可以访问rasa core中大部分功能，如训练，处理消息，加载对话模型，获取下一个动作以及处理channel。    

	from rasa_core.agent import Agent
	from rasa_core.interpreter import RasaNLUInterpreter
	interpreter = RasaNLUInterpreter( "models/ids/nlu")
	agent = Agent.load("models/current/dialogue",  interpreter=interpreter)
	try:
	    ans=agent.handle_text("hello")
	    #ans=agent.handle_message("hello")
	    ans.send(None)
	except StopIteration as e:
	    print(e.value)
 
参考：https://rasa.com/docs/core/api/agent/	    
  
## Interpreters    
NLU文本解释器，有RasaNLUHttpInterpreter和RasaNLUInterpreter两种，  
  
## Event  
对话系统能够处理和支持的所有事件的列表，大多数情况下以json格式使用或接受event    
1. 设置插槽事件    
在tracker上设置插槽的事件

		{  
		    'event': 'slot',  
		    'name': 'departure_airport',  
		    'value': 'BER',  
		}    
2. 重置会话事件    
重置tracker上记录的任何内容。

		{  
		    'event': 'restart'  
		}    
3. 重置插槽事件    
重置所有插槽内容

		{  
		    'event': 'reset_slots'  
		}    
4. 设置提醒事件    
设置将来要执行的操作。

		{  
		    'event': 'reminder',  
		    'action': 'my_action',  
		    'date_time': '2018-09-03T11:41:10.128172',  
		    'name': 'my_reminder',  
		    'kill_on_user_msg': True  
		}    
5. 暂停对话    
阻止机器人响应消息。 行动预测将暂停，直至resumed。

		{  
		    'event': 'pause',  
		}    
6. 恢复对话    
恢复以前暂停的对话。 机器人将开始再次预测行动。

		{  
		    'event': 'resume',  
		}    
7. 强制采取后续行动    
强制下一个动作是固定的动作，而不是预测下一个动作。

		{  
		    'event': 'followup',  
		    'name': 'my_action'  
		}    
8. 自动跟踪事件    
* 用户发送消息--用户发送消息给bot

		{  
		    'event': 'user',  
		    'text': 'Hey',  
		    'parse_data': {  
		        'intent': {'name': 'greet', 'confidence': 0.9},  
		        'entities': []  
		    }  
		}      
* bot回复消息给用户

		{  
		    'event': 'bot',  
		    'text': 'Hey there!',  
		    'data': {}  
		}    
* 撤回用户消息--撤消在最后一条用户消息（包括消息的用户事件）之后发生的所有副作用。

		{  
		    'event': 'rewind'  
		}    
* 撤销action--撤消在最后一个动作之后发生的所有副作用（包括动作的动作事件）

		{  
		    'event': 'undo',  
		}    
* 记录已执行的操作

		{  
		    'event': 'action',  
		    'name': 'my_action'  
		}    
  
## Action  
操作是机器人响应用户输入而运行的内容。 Rasa Core中有三种操作：    
1. default actions (action_listen, action_restart, action_default_fallback)    
2. utter actions，以utter_开头，只是向用户发送消息    
3. custom actions，任何其他动作，这些动作都可以运行任意代码    
  
### Utter Actions  
定义在domain.yml文件中，以utter_开头，例如：    
  
	templates:  
	  utter_my_message:  
	    - "this is what I want my action to say!"    
  
### Custom Actions  
操作可以运行您想要的任何代码。 自定义操作可以打开灯光，向日历添加事件，检查用户的银行余额或您可以想象的任何其他内容    
当预测到自定义操作时，Core将调用您可以指定的端点。 此端点应该是响应此调用的Web服务器，运行代码并可选地返回信息以修改对话状态。    
action server定义在endpoints.yml中，例如：    
  
	action_endpoint:  
	  url: "http://localhost:5055/webhook"    
在rasa core启动脚本rasa_core.run中添加--endpoints endpoints.yml参数。    
可以用node.js，.NET，java或任何其他语言创建一个动作服务器，并在那里定义你的动作 - 但是rasa提供了一个小的python sdk来使开发变得更加容易。    
#### 使用python完成自定义action（rasa_core_sdk）    
安装rasa_core_sdk    
  
	pip install rasa_core_sdk    
如果action实现在名为actions.py的文件中定义，运行以下命令：    
  
	python -m rasa_core_sdk.endpoint --actions actions      
action.py示例如下：    
  
	from rasa_core_sdk import Action  
	from rasa_core_sdk.events import SlotSet  
	class ActionCheckRestaurants(Action):  
	   def name(self):  
	      # type: () -> Text  
	      return "action_check_restaurants"  <!--“action_check_restaurants”名称必须在domain.yml文件的actions域定义-->  
	  
	   def run(self, dispatcher, tracker, domain):  
	      # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict[Text, Any]]  
	      cuisine = tracker.get_slot('cuisine')  
	      q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)  
	      result = db.query(q)  
	      return [SlotSet("matches", result if result is not None else [])]    
  
> Action.run(dispatcher, tracker, domain)[source]    
> action实际执行函数    
> 参数：    
> dispatcher (CollectingDispatcher) :给用户发送信息，使用dipatcher.utter_message()或rasa_core_sdk.executor.CollectingDispatcher方法。    
> tracker (Tracker)：状态跟踪器，可以使用tracker.get_slot(slot_name)访问插槽值、最新的用户消息（tracker.latest_message.text）和任何其他rasa_core_sdk.Tracker属性    
> domain (Dict[Text, Any]) – the bot’s domain     
> return:    
> List[Dict[Text, Any]]；rasa_core_sdk.events.Event通过endpoint返回的实例    
  
#### 使用其他语言实现action  
参考https://rasa.com/docs/core/customactions/    
  
### Default Actions  
rasa中有三种默认action：    
action_listen：等待用户输入    
action_restart：重置整个会话，通常由使用/ restart触发    
action_default_fallback：撤消最后一条用户消息（就好像用户没有发送它）并发出机器人不理解的消息    
默认action可以被修改，将action名称添加到domain中的actions列表中，rasa core会将其视为自定义action，如：    
  
    actions:  
    - action_listen     
  
#### action_default_fallback    
有时候你想回到后退行动，比如说“对不起，我不明白”。 为此，将FallbackPolicy添加到策略集合中。 如果意图识别的置信度低于nlu_threshold，或者没有任何对话策略预测置信度高于core_threshold，则执行回退操作。    
在policy.yml中配置FallbackPolicy，并添加阈值和fallback action    
  
	policies:  
	  - name: "FallbackPolicy"  
	    # min confidence needed to accept an NLU prediction  
	    nlu_threshold: 0.3  
	    # min confidence needed to accept an action prediction from Rasa Core  
	    core_threshold: 0.3  
	    # name of the action to be called if the confidence of intent / action  
	    # is below the threshold  
	    fallback_action_name: 'action_default_fallback'    
使用python实现fallback    
  
	from rasa_core.policies.fallback import FallbackPolicy  
	from rasa_core.policies.keras_policy import KerasPolicy  
	from rasa_core.agent import Agent  
	  
	fallback = FallbackPolicy(fallback_action_name="action_default_fallback",  
	                          core_threshold=0.3,  
	                          nlu_threshold=0.3)	  
	agent = Agent("domain.yml", policies=[KerasPolicy(), fallback])  
		  
## 插槽（slot）  
插槽是机器人的记忆。 它们充当key-value存储器，其可用于存储用户提供的信息（例如，他们的家乡）以及关于外部世界收集的信息（例如，数据库查询的结果）。    
插槽是用来实现多轮对话，不同的行为有不同的插槽类型。     
**policy没有权限访问插槽具体内容，只能查看插槽是否被设置。**   
  
### [插槽类型](https://rasa.com/docs/core/api/slots_api/#slot-types)    
Text Slot：type: text，文本类型，用户只需要关心是否被指定  
Boolean Slot：type: bool，True or False    
Categorical Slot：type: categorical，指定列表中的一个    
Float Slot：type: float，浮点数据    
List Slot：type: list，存储在插槽中的列表长度不会影响对话。    
Unfeaturized Slot：type: unfeaturized，用于存储数据，不影响对话      
  
#### 自定义插槽类型    
在下面的代码中，我们定义了一个名为NumberOfPeopleSlot的槽类。 特征定义了如何将此槽的值转换为我们的机器学习模型可以处理的向量。 我们的插槽有三个可能的“值”，我们可以用长度为2的向量表示    
  
		(0,0)	not yet set  
		(1,0)	between 1 and 6  
		(0,1)	more than 6    
  
	from rasa_core.slots import Slot  
	  
	class NumberOfPeopleSlot(Slot):  
	  
	    def feature_dimensionality(self):  
	        return 2  
	  
	    def as_feature(self):  
	        r = [0.0] * self.feature_dimensionality()  
	        if self.value:  
	            if self.value <= 6:  
	                r[0] = 1.0  
	            else:  
	                r[1] = 1.0  
	    return r    
现在我们还需要一些训练stories，以便Rasa Core可以从中学习如何处理不同的情况：

	# story1  
	...  
	* inform{"people": "3"}  
	- action_book_table  
	...  
	# story2  
	* inform{"people": "9"}  
	- action_explain_table_limit  
  
### 设置插槽  
1. 设置默认值

    	slots:  
    	  name:  
    		type: text  
    		initial_value: "human"    
2. NLU中设置插槽    
如果NLU模型选择了一个实体，并且您的域包含一个具有相同名称的插槽，则会自动设置该插槽。

	    # story_01  
	    * greet{"name": "Ali"}  
	      - slot{"name": "Ali"}  #不用设置slot，rasa会自动关联    
	      - utter_greet  
3. 通过点击按钮设置插槽    
您可以使用按钮作为快捷方式。 Rasa Core将以/开头的消息发送到RegexInterpreter，它希望NLU输入的格式与story文件相同，例如： /intent{entities}。 例如，如果您让用户通过单击按钮选择颜色，则按钮有效负载可能是/choose{"color": "blue"}和/choose{“color”：“red”}。    
可以在此domain文件中指定此项：

		utter_ask_color:  
		- text: "what color would you like?"  
		  buttons:  
		  - title: "blue"  
		    payload: '/choose{"color": "blue"}'  
		  - title: "red"  
		    payload: '/choose{"color": "red"}'    
4. action中设置插槽    
在这种情况下，stories中需要包含插槽，例如，您有一个自定义操作来获取用户的个人资料，并且您有一个名为account_type的分类插槽。 运行fetch_profile操作时，它会返回rasa_core.events.SlotSet事件：    
  
		#domain.yml  
		slots:  
		   account_type:  
		      type: categorical  
		      values:  
		      - premium  
		      - basic  
  
		#action.py  
		from rasa_core_sdk.actions import Action  
		from rasa_core_sdk.events import SlotSet  
		import requests  
		  
		class FetchProfileAction(Action):  
		    def name(self):  
		        return "fetch_profile"  
		  
		    def run(self, dispatcher, tracker, domain):  
		        url = "http://myprofileurl.com"  
		        data = requests.get(url).json  
		        return [SlotSet("account_type", data["account_type"])]    
  
		stories.md  
		# story_01  
		* greet  
		  - action_fetch_profile  
		  - slot{"account_type" : "premium"}  
		  - utter_welcome_premium  
		  
		# story_02  
		* greet  
		  - action_fetch_profile  
		  - slot{"account_type" : "basic"}  
		  - utter_welcome_basic    
在这种情况下，您必须在故事中包含 -  slot {}部分。 Rasa Core将学习如何使用此信息来决定采取的正确操作（在本例中为utter_welcome_premuim或utter_welcome_basic）。    
  
### 填槽过程  
如果您需要连续收集多条信息，我们建议您创建一个FormAction。 这是一个单一操作，其中包含循环所需插槽的逻辑，并询问用户此信息。 在Rasa Core的examples / formbot目录中有一个完整的示例使用表单    
https://rasa.com/docs/core/slotfilling/#id7  
  
## [Responses管理](https://rasa.com/docs/core/responses/)  
如果您希望机器人响应用户消息，则需要管理机器人响应。    
Responses管理有两种方法：    
1. 在domain文件中包含response    
在domain文件中，添加一下内容，这种方式管理response，每次修改都需要重新训练rasa core才能生效。

		templates:  
		  utter_greet:  
		    - "hey there {name}!"  # {name} will be filled by slot (same name) or by custom action code  
2. 使用外部服务生成响应    
如果机器人想要向用户发送消息，它将使用POST请求调用外部HTTP服务器。需要在endpoints.yml配置服务器相关信息。

		 nlg:  
		  url: http://localhost:5055/nlg  
  
## Max History  
定义在policy.yml文件中，该参数控制模型查看的对话历史记录，以决定接下来要采取的操作。    
假设你有一个out_of_scope意图来描述偏离主题的用户消息。 如果机器人连续多次看到此意图，您可能想告诉用户您可以帮助他们做什么。 所以你的story可能如下：

	* out_of_scope  
	   - utter_default  
	* out_of_scope  
	   - utter_default  
	* out_of_scope  
	   - utter_help_message    
  
要让Rasa Core学习这种模式，max_history必须至少为3。    
  
## policy    
在policy中定义不同的策略，并且可以在单个rasa_core.agent.Agent中包含多个策略，在每一轮对话，都将使用以最高置信度预测下一个动作的策略。    
处理在policy.yml文件中定义策略，还可以使用python代码定义策略，如：    
  
	from rasa_core.policies.memoization import MemoizationPolicy  
	from rasa_core.policies.keras_policy import KerasPolicy  
	from rasa_core.agent import Agent  
	  
	agent = Agent("domain.yml",  
	               policies=[MemoizationPolicy(), KerasPolicy()])    
**Memoization Policy：**MemoizationPolicy只记忆训练数据中的对话。 如果训练数据中存在这种确切的对话，它会自信地预测下一个动作1.0，否则它会以置信度0.0预测无。    
**Keras Policy：**KerasPolicy使用Keras中实现的神经网络来选择下一个动作。 默认体系结构基于LSTM，但您可以覆盖KerasPolicy.model_architecture方法以实现您自己的结构。    
**Embedding Policy:**经常性嵌入对话策略（REDP）参考论文https：//arxiv.org/abs/1811.11707    
**Two-stage Fallback Policy：**此策略处理多个阶段的低NLU置信度。      
  
### 策略比较  
rasa提供比较两个策略脚本，创建多个不同的策略文件，使用如下脚本进行比较训练：    
  
	python -m rasa_core.train compare -c policy_config1.yml policy_config2.yml \  
	  -d domain.yml -s stories_folder -o comparison_models --runs 3 --percentages \  
	  0 5 25 50 70 90 95    
对于提供的每个策略配置，Rasa Core将进行多次训练，将0,5,25,50,70和95％的训练story排除在训练数据之外。 多次运行，以确保结果一致。    
训练完成后，可以在比较模式下使用评估脚本来评估刚训练的模型：    
  
	python -m rasa_core.evaluate compare --stories stories_folder \  
	  --core comparison_models \  
	  -o comparison_results    
这将评估训练集上的每个模型，并绘制一些图表以显示哪个策略最佳。 通过评估整套故事，可以衡量Rasa Core对预测故事的预测效果。    
  
## 可视化stories（Visualizing your Stories）  
	cd examples/concertbot/  
	python -m rasa_core.visualize -d domain.yml -s data/stories.md -o graph.html -c config.yml     
	使用特定数据可视化数据流动流程：     
	添加--nlu_data mydata.json  
	  
	visualize.py  
	from rasa_core.agent import Agent  
	from rasa_core.policies.keras_policy import KerasPolicy  
	from rasa_core.policies.memoization import MemoizationPolicy  
	  
	if __name__ == '__main__':  
	    agent = Agent("domain.yml",  
	                  policies=[MemoizationPolicy(), KerasPolicy()])  
	  
	    agent.visualize("data/stories.md",  
	                    output_file="graph.html", max_history=2)  
		  
  
  
