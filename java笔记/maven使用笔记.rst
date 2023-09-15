maven使用笔记
============================

下载
------------
下载地址：http://maven.apache.org/download.cgi

下载zip后缀的文件

安装
--------------
1. 解压apache-maven-3.8.6.zip
2. 配置环境变量，修改~/.bashrc
   ::
    
    export MAVEN_HOME=/usr/local/apache-maven-3.8.6
    export PATH=$PATH:$MAVEN_HOME/bin

3. source ~/.bashrc

配置
--------------
找到maven的安装路径下的conf/settings.xml文件；配置好localRepository(本地仓库)标签的路径

.. code-block:: xml

     <localRepository>/mydata/maven/repository</localRepository>

配置镜像源地址

.. code-block:: xml

    <mirror>
      <id>alimaven</id>
      <name>aliyun maven</name>
       <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
      <mirrorOf>central</mirrorOf>
    </mirror>

执行命令: mvn help:system查看是否配置成功

创建maven工程
--------------------
1. 创建maven项目文件，在终端执行mvn archetype:generate，它会联网自动下载一些需要的插件文件，然后要求选择项目的类型
2. 按了enter选择默认（默认是7: org.apache.maven.archetypes:maven-archetype-quickstart ，包含maven工程样例。
3. 然后到输入groupId、artifactId、version和package
   ::

    groupId:项目属于哪个组，举个例子，如果你的公司是mycom，有一个项目为myapp，那么groupId就应该是com.mycom.myapp. 
    artifactId:当前maven项目在组中唯一的ID,比如，myapp-util,myapp-domain,myapp-web等。 
    version:版本号
    package:包名，如com.test
4. 最后enter键确定，提示创建成功
5. 切换目录到artifactId目录下，查看项目文件

maven命令
------------------
.. code-block:: shell

  # 编译项目,生成class
  mvn compile
  # 把编译的class文件清除掉
  mvn clean
  # 运行测试
  mvn test
  # 生成jar包，并把jar包存储到maven管理的repository目录下了
  mvn install
  # 运行单个java文件
  # 不需要传递参数
  mvn exec:java -Dexec.mainClass="com.jsoft.test.MainClass" 
  #需要传递参数
  mvn exec:java -Dexec.mainClass="com.jsoft.test.MainClass" -Dexec.args="arg0 arg1 arg2" 
  # 指定对classpath的运行时依赖
  mvn exec:java -Dexec.mainClass="com.jsoft.test.MainClass" -Dexec.classpathScope=runtime 

