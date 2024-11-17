

def mistral_prompt(question, context, task = "QA"):
    if task == "QA":
        instruction = "You are a helpful assistant. Your task is answer the question using the given context."
    
    prompt = (
        f"[INST] {instruction}\n\n"
        f"Question : {question}\n\n"
        f"Context : {context}\n\n"
        f"Please provide your response. [/INST]"
        )
    return prompt

