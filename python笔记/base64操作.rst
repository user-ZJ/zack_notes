base64操作
=========================

文件转base64
------------------------------
.. code-block:: python

    with open(filepath,'rb') as f:
        filebytes = f.read()
        filebase64 = base64.b64encode(filebytes).decode('utf-8')

base64转byte
---------------------
.. code-block:: python

    filebytes = base64.b64decode(base64_str.encode())