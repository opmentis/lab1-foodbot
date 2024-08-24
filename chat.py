import requests
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
import typer

# Set LLM cache
set_llm_cache(InMemoryCache())

# Initialize Typer app
app = typer.Typer()

def openai_chat(apimessage, openai_api_key: str):
    if not openai_api_key:
        raise ValueError("OpenAI API key is required.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    if not apimessage:
        apimessage = ['Hello']

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": apimessage[0]}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise ValueError(f"Error: {response.status_code} - {response.text}")


def gpt4all_chat(apimessage):
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.chains import LLMChain
    from langchain_community.llms import GPT4All
    from langchain_core.prompts import PromptTemplate

    if not apimessage:
        apimessage = 'Hello'

    template = """
    You are a helpful assistant.
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    local_path = "models/gpt4all-13b-snoozy-q4_0.gguf"
    callbacks = [StreamingStdOutCallbackHandler()]

    llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.invoke(apimessage)


def initiate_chat(wallet_address: str, function):
    response_messages = ["Hello"]
    try:
        while True:
            chat_message = function(apimessage=response_messages)

            if isinstance(chat_message, dict):
                chat_message = chat_message.get("text", "")

            response = requests.post("http://54.74.133.71/chat", json={"wallet_address": wallet_address, "prompt": chat_message})

            if response.status_code == 200:
                response_json = response.json()
                response_messages = [item["response"] for item in response_json if item is not None]
                if response_messages:
                    print("Response:", response_messages[0])
            else:
                print("Failed to send chat message, status code:", response.status_code)
                break

    except Exception as e:
        print("Error:", e)
        raise RuntimeError(f"Chat session failed: {str(e)}")

    return {"message": "Chat session ended", "response": response_messages}


@app.command()
def chat(function: str, wallet_address: str, api_key: str = typer.Option(None, help="OpenAI API key for Openai function")):
    """
    Initiates a chat session with the specified function and wallet address.

    Args:
        function (str): The model to use ('Openai' or 'Gpt4all').
        wallet_address (str): The wallet address to associate with the chat.
        api_key (str, optional): The OpenAI API key. Required for the Openai function.
    """
    if function == "Openai":
        if not api_key:
            raise typer.BadParameter("OpenAI API key is required for the Openai function.")
        initiate_chat(wallet_address, function=lambda apimessage: openai_chat(apimessage, openai_api_key=api_key))
    elif function == "Gpt4all":
        initiate_chat(wallet_address, function=gpt4all_chat)
    else:
        raise typer.BadParameter("Function must be 'Openai' or 'Gpt4all'.")


if __name__ == "__main__":
    app()
