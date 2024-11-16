from mistralai import Mistral

def get_llm_response(query, context=None):
    api_key = 'gTJxw3zjDDXf4UfB0mvPQ4wKYxeNEtd2'
    model = "mistral-large-latest"
    mistral_client = Mistral(api_key=api_key)

    messages = []
    if context:
        messages.append({"role": "system", "content": context,})
    messages.append({"role": "user", "content": query,})
    
    chat_response = mistral_client.chat.complete(
        model = model,
        messages = messages
    )
    
    return chat_response.choices[0].message.content