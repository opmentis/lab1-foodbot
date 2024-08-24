# Food Bot 

This project implements a chatbot using various natural language processing models. Below are the instructions for running the scripts and information about the available models.

## Requirements

- Python 3.9 or higher
- For OpenAI model: OpenAI API key (available at www.openai.com)
- For Gpt4all model: Minimum 16GB RAM and 6.86GB of storage
- For Llama model: GPU and minimum 16GB storage

## Setup

1. Clone this repository to your local machine:

   ```bash
   cd cloned-repo
2. To install the required Python packages, follow these steps:

Without downloading the Llama model:
If you prefer not to download the Llama model, simply run:
```bash
pip install -r requirements.txt
```
This will install the necessary packages listed in the requirements.txt file.

With Gpt4all model download (Mac and Ubuntu users):
For users who wish to download the Gpt4all model and set up shell completion, execute the setup.sh script. This will handle the installation of required dependencies and download the default Gpt4all model:
```bash
./setup.sh
```
Ensure that the script has executable permissions before running it. If you encounter a "permission denied" error, run the following command:
```bash
chmod +x setup.sh
```
Then, rerun the script:
```bash
./setup.sh
```
3. Downloading the Gpt4all model separately:
If you only want to download the Gpt4all model without setting up the other environment configurations, you can run the download_model.sh script:
```bash
./download_model.sh
```

As with the setup.sh script, ensure the download_model.sh script has executable permissions:

```bash
chmod +x download_model.sh
./download_model.sh
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

Run the `chat.py` script with the appropriate arguments to initiate the chatbot:

```bash
cd to lab1-foodbot

run the below help to see what you need to start the script
python chat.py --help

python chat.py [Function_Name] [Wallet_Address] --api_key [OpenAI_API_Key]


python chat.py Openai 0xYourWalletAddress --api_key sk-YourOpenAIAPIKey


python chat.py Gpt4all 0xYourWalletAddress
```


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



