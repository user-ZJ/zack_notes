nginx使用
===========================

nginx配置
----------------------
.. code-block:: yaml

    # 定义一个名为myapp的upstream，包含两台后端服务器
    upstream Qwen2-72B-Instruct {
        server 192.168.68.103:9997;
        server 192.168.68.99:9997;
    }

    upstream Qwen2-7B-Instruct {
        server 192.168.68.104:9906;
    }

    upstream glm-4-9b-chat {
        server 192.168.68.104:9902;
        server 192.168.68.104:9905;
    }

    server {
        listen 80;

        # 将所有请求代理到定义的upstream
        location ~ /v1/chat/completions|/v1/completions {
            if ($http_model_name = "Qwen2-72B-Instruct") {
            proxy_pass http://Qwen2-72B-Instruct;
            }
            if ($http_model_name = "Qwen2-7B-Instruct") {
            proxy_pass http://Qwen2-7B-Instruct;
            }
            if ($http_model_name = "glm-4-9b-chat") {
            proxy_pass http://glm-4-9b-chat;
            }
            if ($http_model_name = "cnjy-cut-glm4-lora") {
            proxy_pass http://glm-4-9b-chat;
            }
            if ($http_model_name = "cnjy-level-glm4-lora") {
            proxy_pass http://glm-4-9b-chat;
            }
            proxy_pass http://Qwen2-7B-Instruct;
        }
    }


配置docker
---------------------------
.. code-block:: docker

    FROM nginx:1.17
    ENV TZ=Asia/Shanghai

    COPY nginx.conf /etc/nginx/conf.d/default.conf
    EXPOSE 80





