# Food Bot 

This project implements a chatbot using various natural language processing models. Below are the instructions for running the scripts and information about the available models.

## Requirements

- Python 3.9 or higher
- For OpenAI model: OpenAI API key (available at www.openai.com)
- For Gpt4all model: Minimum 16GB RAM and 6.86GB of storage
- For Llama model: GPU and minimum 16GB storage

## Setup

1. Clone this repository to your local machine.

2. To install the required Python packages, follow these steps:

If you prefer not to download the llama model, simply run ```pip install -r requirements.txt``` to install the necessary packages from the requirements file.
For Mac and Ubuntu users who wish to download the llama model, execute the ```setup.sh``` bash script which uses the Gpt4all script for installation.

 to install required dependencies and download the default Gpt4all model:

3. Install the Opmentis package:
To facilitate user registration functionalities within the chatbot, install the opmentis package using pip:
```bash
pip install opmentis
```
# Usage

### Registering a Miner
To register a new user as a miner:

```python
from opmentis import register_miners

# Example: Registering a miner
miner_wallet_address = "miner_wallet_address"
miner = register_miners(wallet_address=miner_wallet_address)
print("Miner Registered:", miner)
```
### Check your data
To check miners data:

```python
from opmentis import userdata, endchat

# Example: check miners data
print(endchat())
miner_wallet_address = "miner_wallet_address"
data = userdata(wallet_address=miner_wallet_address)
print(data)

```
4. If you want to end the chat and start a new session whenever you want by handling a signal(ctr+c), you can include the following code in chat.py:

```python
import signal
import sys

def signal_handler(sig, frame):
    endchat()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
```
Alternatively, you can use this code in a separate script to start a new chat session. This way, the chat will end and a new session will start whenever the script is called 

```python
from opmentis import  endchat, userdata


print(endchat())
```

5. If you're using a Gpt4all model other than the default one, download the model file from https://gpt4all.io/index.html and place it in the `models` folder.



## Running the Scripts



### 1. Python Script

Run the `chat.py` script with the appropriate arguments to initiate the chatbot: ```cd to lab1-foodbot``` folder 

python chat.py --function [Function_Name] --api_key [OpenAI_API_Key] --wallet_address [Wallet_Address]

python chat.py --function Gpt4all --wallet_address 55walletaddress




Replace `[Function_Name]` with one of the available models: `Openai`, `llama`, or `Gpt4all`. Provide the OpenAI API key if using the OpenAI model.

## Available Models

### 1. OpenAI Model

- **Description:** State-of-the-art natural language processing model by OpenAI.
- **Requirements:** OpenAI API key.
- **How to Obtain:** Visit www.openai.com to register and obtain the API key.

### 2. Gpt4all Model

- **Description:** A powerful GPT-based language model.
- **Requirements:** Minimum 16GB RAM and 6.86GB of storage.
- **How to Obtain:** Download additional models from https://gpt4all.io/index.html and place them in the `models` folder.

### 3. Llama Model

- **Description:** A language model suitable for conversational agents.
- **Requirements:** GPU and minimum 16GB storage.
- **How to Obtain:** For access and installation instructions, refer to the Llama repository.



