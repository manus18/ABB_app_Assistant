def run_query(agent, query):
    return agent.invoke({"messages": [{"role": "user", "content": query}]})
