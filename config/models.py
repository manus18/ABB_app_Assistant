from langchain_aws import ChatBedrock, BedrockEmbeddings

def get_llm():
    return ChatBedrock(model="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

def get_embeddings():
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")
