# json操作

JSON在python中分别由list和dict组成  

python有两个模块可以对json操作：  

* Json模块提供了四个功能：dumps、dump、loads、load  

* pickle模块提供了四个功能：dumps、dump、loads、load  
  区别：json只能把常用的数据类型序列化（列表、字典、列表、字符串、数字、），比如日期格式、类对象！josn就不行了。而pickle可以序列化所有的数据类型，包括类，函数都可以序列化  

* dumps：数据类型转换成字符串  

* dump：数据类型转换成字符串并存储在文件中  

* loads：字符串转换成数据类型  

* load：把文件打开从字符串转换成数据类型

实例：

    test_dict = {'bigberg': [7600, {1: [['iPhone', 6300], ['Bike', 800], ['shirt', 300]]}]}  
    json_str = json.dumps(test_dict)
    new_dict = json.loads(json_str)
    
    将数据写入json文件中
    with open("../config/record.json","w", encoding='UTF-8') as f:
        json.dump(new_dict,f,ensure_ascii=False)  #dump中文需要加ensure_ascii=False参数
    
    把文件打开，并把字符串变换为数据类型
    with open("../config/record.json",'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
    load_dict['smallberg'] = [8200,{1:[['Python',81],['shirt',300]]}]
    
    with open("../config/record.json","w", encoding='UTF-8') as dump_f:
        json.dump(load_dict,dump_f,ensure_ascii=False,indent=4) #dump中文需要加ensure_ascii=False参数  
            # indent=4，缩进的空格数，设置为非零值时，就起到了格式化的效果，比较美观