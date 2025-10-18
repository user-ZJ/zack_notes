git使用笔记
===================

1. 查看单个文件的提交记录
    ``git log filename``
2. 查看当个文件每次提交的修改
    ``git log -p filenam``
3. 查看最近的n条修改
   ``git log -n``
4. 查看远程仓库
   ``git remote -v``
5. 删除本地分支
   ``git branch -d localBranchName``
6. 删除远程分支
   ``git push origin --delete remoteBranchName``
7. 删除commit
   
   .. code-block:: shell

    # 不删除工作空间改动代码，撤销commit，不撤销git add . 
    git reset --soft HEAD^ 
    # 不删除工作空间改动代码，撤销commit，并且撤销git add . 操作;和 git reset HEAD^ 效果是一样的
    git reset --mixed HEAD^
    # 删除工作空间改动代码，撤销commit，撤销git add . 
    git reset --hard HEAD^
    # 修改commit注释
    git reset --amend

8. 修改远程url

    git remote set-url origin [url]
    例如：git remote set-url origin https://ddns.togit.cc:7777/rffanlab/tensorflow_get_started_doc_cn.git

9. 合并分支

.. code-block:: shell

    git checkout 目标分支
    git merge 待合并分支


打patch
----------------

生成patch
`````````````````
.. code-block:: shell

    # 把所有的修改文件打成 patch
    git diff > test.patch
    git format-patch HEAD^       #生成最近的1次commit的patch
    git format-patch HEAD^^      #生成最近的2次commit的patch
    git format-patch HEAD^^^     #生成最近的3次commit的patch
    git format-patch HEAD^^^^    #生成最近的4次commit的patch
    git format-patch <r1>..<r2>  #生成两个commit间的修改的patch（生成的patch不包含r1. <r1>和<r2>都是具体的commit号)
    git format-patch -1 <r1>     #生成单个commit的patch
    git format-patch <r1>        #生成某commit以来的修改patch（不包含该commit）
    git format-patch --root <r1> #生成从根到r1提交的所有patch


应用patch
`````````````````
* git am会直接将patch的所有信息打上去，而且不用重新git add和git commit，author也是patch的author而不是打patch的人。
* git apply是另外一种打patch的命令，其与git am的区别是：git apply并不会将commit message等打上去，打完patch后需要重新git add和git commit。

.. code-block:: shell

    git apply --stat 0001-limit-log-function.patch  # 查看patch的情况
    git apply --check 0001-limit-log-function.patch # 检查patch是否能够打上，如果没有任何输出，则说明无冲突，可以打上

    git apply xxx.patch
    git am 0001-limit-log-function.patch           # 将名字为0001-limit-log-function.patch的patch打上
    # 添加-s或者--signoff，还可以把自己的名字添加为signed off by信息，作用是注明打patch的人是谁，因为有时打patch的人并不是patch的作者
    git am --signoff 0001-limit-log-function.patch 
    git am ~/patch-set/*.patch                     # 将路径~/patch-set/*.patch 按照先后顺序打上
    git am --abort                                 # 当git am失败时，用以将已经在am过程中打上的patch废弃掉(比如有三个patch，打到第三个patch时有冲突，那么这条命令会把打上的前两个patch丢弃掉，返回没有打patch的状态)
    git am --resolved                              # 当git am失败，解决完冲突后，这条命令会接着打patch


打patch冲突解决
`````````````````````
**方案1（推荐）**

(1) 根据git am失败的信息，找到发生冲突的具体patch文件，然后用命令git apply --reject <patch_name>，强行打这个patch，发生冲突的部分会保存为.rej文件（例如发生冲突的文件是a.txt，那么运行完这个命令后，发生conflict的部分会保存为a.txt.rej），未发生冲突的部分会成功打上patch
(2) 根据.rej文件，通过编辑该patch文件的方式解决冲突
(3) 废弃上一条am命令已经打了的patch：git am --abort
(4) 重新打patch： `git am ~/patch-set/*.patchpatch`

**方案2**

(1) 根据git am失败的信息，找到发生冲突的具体patch文件，然后用命令git apply --reject <patch_name>，强行打这个patch，发生冲突的部分会保存为.rej文件（例如发生冲突的文件是a.txt，那么运行完这个命令后，发生conflict的部分会保存为a.txt.rej），未发生冲突的部分会成功打上patch
(2) 根据.rej文件，通过编辑发生冲突的code文件的方式解决冲突
(3) 将该patch涉及到的所有文件（不仅仅是发生冲突的文件）通过命令git add <file_name>添加到工作区中
(4) 告诉git冲突已经解决，继续打patch: git am --resolved (git am --resolved 和 git am --continue是一样的)


git status中文乱码
-------------------------------
解决方法：

.. code-block:: shell

    git config --global core.quotepath false



git tag 
-------------------------
.. code-block:: shell

    # 在当前commit打tag
    git tag v1.0 
    # 在指定提交打tag
    git  tag v0.9 471fd27
    # 创建带有说明的标签，用-a指定标签名，-m指定说明文字
    git tag -a v0.1 -m "version 0.1 released push url" d5a65e9
    # 查看所有标签
    git tag
    # 查看说明文字
    git show v0.1
    # push单个tag
    git push origin tag_20170908
    # push所有tag
    git push origin --tags


将远程仓库同步到本地仓库
----------------------------
.. code-block:: shell

    mkdir sync && cd sync
    git init
    git remote add github GITHUB_REPO_URL
    git remote add gitlab GITLAB_REPO_URL
    #获取github的SOURCE_BRANCH分支，并push到gitlab的TARGET_BRANCH分支
    git fetch --no-tags github SOURCE_BRANCH
    git pull -u gitlab github/SOURCE_BRANCH:refs/heads/TARGET_BRANCH
    #验证
    git ls-remote gitlab refs/heads/TARGET_BRANCH
