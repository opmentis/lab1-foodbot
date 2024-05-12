from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp


def load_llama_model():
    model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_basename = "llama-2-7b-chat.Q5_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    
    llm = LlamaCpp(
        model_path=model_path,
        max_tokens=256,
        n_gpu_layers=1,
        n_batch=512,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        n_ctx=1024,
        verbose=True,
    )
    
    return model_path, llm

model_path, llm = load_llama_model()


