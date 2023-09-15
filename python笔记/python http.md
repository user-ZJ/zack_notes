# [python http请求](https://www.cnblogs.com/fiona-zhong/p/10179421.html)  
## urllib.request  
1. 不带参数的get请求  
	response = urllib.request.urlopen(url,data=None, [timeout, ]*)  
	返回的response是一个http.client.HTTPResponse object  
	response操作：  
	a) response.info() 可以查看响应对象的头信息,返回的是http.client.HTTPMessage object  
	b) getheaders() 也可以返回一个list列表头信息  
	c) response可以通过read(), readline(), readlines()读取，但是获得的数据是二进制的所以还需要decode将其转化为字符串格式。  
	d) getCode() 查看请求状态码  
	e) geturl() 获得请求的url  
	import urllib  
	__author__='zack'  
	url = 'https://fanyi.baidu.com/?aldtype=16047#auto/zh'  
	response = urllib.request.urlopen(url)  
	print('response header')  
	print(response.info())  
	print('response body')  
	print(" ".join([line for line in response.read().decode("utf8")]))  
2. 带参数的get请求  
	用urllib下面的parse模块的urlencode方法  
	param = {"param1":"hello", "param2":"world"}  
	param = urllib.parse.urlencode(param)　　　　# 得到的结果为：param2=world&param1=hello  
	url = "?".join([url, param])  
	response = urllib.request.urlopen(url)  
  
	import urllib  
	__author__='zack'  
	url = 'https://fanyi.baidu.com/?aldtype=16047#auto/zh'  
	param = {"param1":"hello", "param2":"world"}  
	param = urllib.parse.urlencode(param)  
	url= "?".join([url,param])  
	print(url)  
	response = urllib.request.urlopen(url)  
3. post请求  
	urllib.request.urlopen()默认是get请求，但是当data参数不为空时，则会发起post请求  
	传递的data需要是bytes格式  
	设置timeout参数，如果请求超出我们设置的timeout时间，会跑出timeout error 异常。  
	param = {"param1":"hello", "param2":"world"}  
	param = urllib.parse.urlencode(param).encode("utf8") # 参数必须要是bytes  
	response = urllib.request.urlopen(url, data=param, timeout=10)  
4. 添加headers  
	通过urllib发起的请求，会有一个默认的header：Python-urllib/version，指明请求是由urllib发出的，所以遇到一些验证user-agent的网站时，我们需要伪造我们的headers  
	伪造headers，需要用到urllib.request.Request对象  
	headers = {"user-agent:"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"}  
	req = urllib.request.Request(url, headers=headers)  
	resp = urllib.request.urlopen(req)  
5. 添加cookie  
	为了在请求的时候，带上cookie信息，需要构造一个opener  
	需要用到http下面的cookiejar模块  
	from http import cookiejar  
	from urllib import request  
	a) 创建一个cookiejar对象  
		cookie = cookiejar.CookieJar()  
	b) 使用HTTPCookieProcessor创建cookie处理器  
		cookies = request.HTTPCookieProcessor(cookie)  
	c) 以cookies处理器为参数创建opener对象  
		opener = request.build_opener(cookies)  
	d) 使用这个opener来发起请求  
		resp = opener.open(url)  
	e) 使用opener还可以将其设置成全局的，则再使用urllib.request.urlopen发起的请求，都会带上这个cookie  
		request.build_opener(opener)  
		request.urlopen(url)  
6. IP代理  
	使用爬虫来爬取数据的时候，常常需要隐藏我们真实的ip地址，这时候需要使用代理来完成  
	IP代理可以使用西刺（免费的，但是很多无效），大象代理（收费）等  
	代理池的构建可以写固定ip地址，也可以使用url接口获取ip地址  
	固定ip：  
	from urllib import request  
	import random  
	ippools = ["36.80.114.127:8080","122.114.122.212:9999","186.226.178.32:53281"]  
	def ip(ippools):  
		cur_ip = random.choice(ippools)  
		proxy = request.ProxyHandler({"http":cur_ip})  # 创建代理处理程序对象  
		opener = request.build_opener(proxy, request.HttpHandler)  # 构建代理  
		request.install_opener(opener) # 全局安装  
		for i in range(5):  
			try:  
				ip(ippools)  
				cur_url = "http://www.baidu.com"  
				resp = request.urlopen(cur_url).read().decode("utf8")  
			excep Exception as e:  
				print(e)  
	使用接口构建IP代理池（这里是以大象代理为例）  
	def api():  
		all=urllib.request.urlopen("http://tvp.daxiangdaili.com/ip/?tid=订单号&num=获取数量&foreign=only")  
		ippools = []  
		for item in all:  
			ippools.append(item.decode("utf8"))  
		return ippools  
7. 爬取数据并保存到本地 urllib.request.urlretrieve()  
	如我们经常会需要爬取一些文件或者图片或者音频等，保存到本地  
	urllib.request.urlretrieve(url, filename)  
8. urllib的parse模块  
	a）urllib.parse.quote()  
		这个多用于特殊字符的编码，如我们url中需要按关键字进行查询，传递keyword='诗经'  
		url是只能包含ASCII字符的，特殊字符及中文等都需要先编码在请求  
	b）urllib.parse.urlencode()  
		这个通常用于多个参数时，帮我们将参数拼接起来并编译，向上面我们使用的一样  
9. urllib.error  
	urllib中主要两个异常，HTTPError，URLError，HTTPError是URLError的子类  
	HTTPError包括三个属性：  
	code：请求状态码  
	reason：错误原因  
	headers：请求报头  

## requests  
requests模块在python内置模块上进行了高度的封装，从而使得python在进行网络请求时，变得更加人性化，使用requests可以轻而易举的完成浏览器可有的任何操作。    
1. get  
	requests.get(url)　　#不带参数的get请求  
	requests.get(url, params={"param1":"hello"})　　# 带参数的get请求，requests会自动将参数添加到url后面  
	
2. post  
	requests.post(url, data=json.dumps({"key":"value"}))  
	
3. 定制头和cookie信息  
	header = {"content-type":"application/json"，"user-agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"}  
	cookie = {"cookie":"cookieinfo"}  
	requests.post(url, headers=header, cookie=cookie)  
	requests.get(url, headers=header, cookie=cookie)  
	
4. 返回对象操作  
	使用requests.get/post后会返回一个response对象，其存储了服务器的响应内容，我们可以通过下面方法读取响应内容  
	resp = requests.get(url)  
	resp.url　　　　　　　　   # 获取请求url  
	resp.encoding　　　　       # 获取当前编码  
	resp.encoding='utf8'    　　# 设置编码  
	resp.text　　    　　　　    # 以encoding解析返回内容。字符串形式的响应体会自动根据响应头部的编码方式进行解码  
	resp.content    　　　　　 # 以字节形式（二进制）返回。字节方式的响应体，会自动解析gzip和deflate压缩  
	resp.json()  　　　　　　  # requests中内置的json解码器。以json形式返回，前提是返回的内容确实是json格式的，否则会报错  
	resp.headers　　　　   　 # 以字典形式存储服务器响应头。但是这个字典不区分大小写，若key不存在，则返回None  
	resp.request.headers　 　# 返回发送到服务器的头信息  
	resp.status_code　　 　   # 响应状态码  
	resp.raise_for_status()　   # 失败请求抛出异常  
	resp.cookies　　　　　　 # 返回响应中包含的cookie  
	resp.history　　　　　　   # 返回重定向信息。我们可以在请求时加上allow_redirects=False 来阻止重定向  
	resp.elapsed　　　　　　 # 返回timedelta，响应所用的时间   
	
5. Session()  
	会话对象，能够跨请求保持某些参数。最方便的是在同一个session发出的所有请求之间保持cookies  
	s = requests.Session()  
	header={"user-agent":""}  
	s.headers.update(header)  
	s.auth = {"auth", "password"}  
	resp = s.get(url)  
	resp1 = s.port(url)  
	
6. 代理  
	proxies = {"http":"ip1", "https":"ip2"}  
	requests.get(url, proxies=proxies)  
	
7. 上传文件  
	requests.post(url, files={"file": open(file, 'rb')})     
	
	requests.post(url, files={"file": open(file, 'rb').read()})
	
	把字符串当做文件上传方式：  
	requests.post(url, files={"file":('test.txt', b'hello world')}})　　# 显示的指明文件名为test.txt  
	
8. 身份认证（HTTPBasicAuth）  
  from requests.auth import HTTPBasicAuth  
  resp = request.get(url, auth=HTTPBasicAuth("user", "password"))  
  另一种非常流行的HTTP身份认证形式是摘要式身份认证  
  requests.get(url, HTTPDigestAuth("user", "password"))  
