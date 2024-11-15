import requests
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
import typer
from typing import Optional
import opmentis

# Set LLM cache
set_llm_cache(InMemoryCache())

app = typer.Typer()

# Global variable to store the authentication token
auth_token = None

# Base URL for the API
BASE_URL = "https://labfoodbot.opmentis.xyz/api/v1"


def authenticate_or_register(wallet_address: str, labid: str, role_type: str, stake: int = None) -> Optional[str]:
    """
    Authenticate or register a user to obtain an authentication token.
    If the user is already registered, a token will be provided without needing the stake.
    Uses the opmentis library to handle the registration and authentication process.
    """
    try:
        # Register the user using the opmentis library
        register_response = opmentis.register_user(wallet_address, labid, role_type)
        
        # Extract the access token from the response
        global auth_token
        auth_token = register_response["registration_response"]["message"]["external_response"].get("access_token")

        if auth_token:
            print(f"Token: {auth_token}")
            return auth_token
        else:
            print("Authentication failed or user not registered.")
            return None

    except Exception as e:
        print(f"An error occurred during authentication or registration: {e}")
        return None

def get_headers():
    """Helper function to get headers with the authorization token."""
    if not auth_token:
        raise ValueError("Auth token is not set. Please authenticate first.")
    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

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
            {"role": "system", "content": "You are a helpful assistant trying to hold a conversation."},
            {"role": "user", "content": apimessage[0]}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print(f"Opmentis Bot: {response.json()['choices'][0]['message']['content']}")
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

            response = requests.post(
                f"{BASE_URL}/chat", 
                json={"wallet_address": wallet_address, "prompt": chat_message}, 
                headers=get_headers()
            )

            if response.status_code == 200:
                response_json = response.json()
                response_messages = [item["response"] for item in response_json if item is not None]
                if response_messages:
                    print("Opmentis User:", response_messages[0])
            else:
                print("Failed to send chat message, status code:", response.status_code)
                break

    except Exception as e:
        print("Error:", e)
        raise RuntimeError(f"Chat session failed: {str(e)}")

    return {"message": "Chat session ended", "response": response_messages}

@app.command("chat")
def chat(
    function: str = typer.Argument(..., help="The model to use ('Openai' or 'Gpt4all')."),
    wallet_address: str = typer.Argument(..., help="The wallet address to associate with the chat."),
    labid: str = typer.Option(..., "--labid", help="The unique ID of the lab."),
    role_type: str = typer.Option(..., "--role-type", help="The role type, e.g., 'miner' or 'validator'."),
    stake: Optional[int] = typer.Option(None, help="Stake amount for new users"),
    api_key: str = typer.Option(None, help="OpenAI API key for Openai function")
):
    """
    Register or authenticate a user and initiate a chat session.

    Args:
        function (str): The model to use ('Openai' or 'Gpt4all').
        wallet_address (str): The wallet address to associate with the chat.
        labid (str): The lab ID.
        role_type (str): The role type ('miner' or 'validator').
        stake (Optional[int]): Stake amount for registration (only required if the user is not registered).
        api_key (str, optional): The OpenAI API key. Required for the Openai function.
    """
    # Step 1: Authenticate or register the user
    token = authenticate_or_register(wallet_address, labid, role_type, stake)
    if not token:
        print("Authentication failed. Unable to proceed with chat.")
        return
    
    # Step 2: Initiate chat based on function type
    if function.lower() == "openai":
        if not api_key:
            raise typer.BadParameter("OpenAI API key is required for the Openai function.")
        initiate_chat(wallet_address, function=lambda apimessage: openai_chat(apimessage, openai_api_key=api_key))
    elif function.lower() == "gpt4all":
        initiate_chat(wallet_address, function=gpt4all_chat)
    else:
        raise typer.BadParameter("Function must be 'openai' or 'gpt4all'.")



if __name__ == "__main__":
    app()
