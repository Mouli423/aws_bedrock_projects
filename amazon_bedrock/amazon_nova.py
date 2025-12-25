import boto3
import json

bedrock=boto3.client(service_name="bedrock-runtime")

prompt="You are an intelligent, supportive and experienced individual in the field of AI, Gen AI and Agenti AI." \
" explain how to get placed in one of the top product-based companies in AI, GenAI and Agentic AI roles."

model_id="amazon.nova-lite-v1:0"

body = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": [
                { "text": prompt }
            ]
        }
    ],
    "inferenceConfig": {
        "maxTokens": 5000,
        "temperature": 0.7,
        "topP": 0.9
    }
})
accept = "application/json"
content_type = "application/json"

response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
response_body = json.loads(response.get("body").read())

print(response_body["output"]["message"]["content"][0]["text"])
