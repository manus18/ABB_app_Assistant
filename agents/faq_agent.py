from langgraph.prebuilt import create_react_agent

def build_agent(llm, tools):
    system_prompt = "You are a helpful ABB product assistant. Answer questions using the retrieved ABB product data and catalogues."
    return create_react_agent(llm, tools, prompt=system_prompt)
