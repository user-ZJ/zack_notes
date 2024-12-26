jsonrpc
========================

源码仓库：https://github.com/smagafurov/fastapi-jsonrpc

示例
---------------
* 服务端

.. code-block:: python

    import fastapi_jsonrpc as jsonrpc
    from pydantic import BaseModel
    from fastapi import Body

    app = jsonrpc.API()
    api_v1 = jsonrpc.Entrypoint('/api/v1/jsonrpc')

    class MyError(jsonrpc.BaseError):
        CODE = 5000
        MESSAGE = 'My error'

        class DataModel(BaseModel):
            details: str


    @api_v1.method(errors=[MyError])
    def echo(
        data: str = Body(..., examples=['123']),
    ) -> str:
        if data == 'error':
            raise MyError(data={'details': 'error'})
        else:
            return data


    app.bind_entrypoint(api_v1)


    if __name__ == '__main__':
        import uvicorn
        # uvicorn.run('example1:app', port=5000, access_log=False)
        uvicorn.run(app, host="0.0.0.0", port=5000)

* 客户端

.. code-block:: python

    import requests
    import json

    url = "http://10.32.1.141:5000/api/v1/jsonrpc"

    payload = json.dumps({
    "jsonrpc": "2.0",
    "method": "echo",
    "params": {
        "data": "Hello, JSON-RPC!"
    },
    "id": 123456
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
