import requests

url = "http://localhost:8000/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer fake-key",
}

data = {
    "model": "NVILA-Lite-8B",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        # "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",
                        "url": "https://resources.chimhaha.net/article/1697450293398-9qc8qommtjf.jpg",
                        # Or you can pass in a base64 encoded image
                        # "url": "data:image/png;base64,<base64_encoded_image>",
                    },
                },
            ],
            # "content": [
            #     {"type": "text", "text": "What's in this video?"},
            #     {
            #         "type": "video_url",
            #         "video_url": {
            #             "url": "https://www.youtube.com/shorts/pmEz6bgVPGI",
            #         },
            #     },
            # ],
        }
    ],
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result["choices"][0]["message"]["content"])
