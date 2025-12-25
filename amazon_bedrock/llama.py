import boto3
import json


prompt="You are a researcher and write a bolg post on future of AI, how it effects the current software industry "

bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")

payload={
    "prompt": prompt,
    "temperature": 0.8,
    "top_p": 0.8,
    "max_gen_len": 500
}

body=json.dumps(payload)

model_id="meta.llama3-70b-instruct-v1:0"

response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get("body").read())
response_text=response_body["generation"]
print(response_text)
