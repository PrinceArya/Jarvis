import sys
import os

# Add parent directory to access model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import warnings
from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Suppress warnings that clutter output
warnings.filterwarnings("ignore")

from model import SmallLanguageModel
from tokenizer import BPETokenizer

app = FastAPI()

# Mount static files folder to serve the index.html and assets
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Instantiate and prepare Model + Tokenizer strictly mimicking the most recent working logic
vocab_size = 4000
dim = 256
n_layers = 12
n_heads = 8
context_length = 128
max_seq_len = context_length * 2
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and tokenizer...")
tokenizer = BPETokenizer()
tokenizer_path = os.path.join(os.path.dirname(__file__), '..', "wikitext_tokenizer")
tokenizer.load(tokenizer_path)

model = SmallLanguageModel(
    vocab_size=vocab_size,
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    max_seq_len=max_seq_len
).to(device)

model_path = os.path.join(os.path.dirname(__file__), '..', "slm_model_epoch_10.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully!")


# Define Langchain Custom LLM Wrapper
class CustomSLM(LLM):
    
    @property
    def _identifying_params(self) -> dict:
        return {"model_name": "ScratchSLM"}

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        max_new_tokens = 40
        tokens = tokenizer.encode(prompt)
        
        # If the history string encodes to more tokens than the model's safe context limit,
        # aggressively trim it from the oldest message backward to avoid breaking RoPE matrices
        max_allowed_context = max_seq_len - max_new_tokens
        if len(tokens) > max_allowed_context:
            tokens = tokens[-max_allowed_context:]
            
        if len(tokens) == 0:
            tokens = [0]
            
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, past_key_values = model(input_ids)
            
        next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
        tokens.append(next_token)
        generated_tokens = [next_token]
        
        start_pos = input_ids.shape[1]
        
        for _ in range(max_new_tokens - 1):
            input_id = torch.tensor([[next_token]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, past_key_values = model(
                    input_id, 
                    start_pos=start_pos, 
                    past_key_values=past_key_values, 
                    causal=False
                )
            next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
            
            tokens.append(next_token)
            generated_tokens.append(next_token)
            start_pos += 1
            
        result = tokenizer.decode(generated_tokens)
        
        # Simple stop word implementation logic for Langchain
        if stop is not None:
            for s in stop:
                if s in result:
                    result = result.split(s)[0]
                    
        return result


# Setup Langchain Conversation Chain Globally for simple local session
# We use k=3 to remember the last 3 turns avoiding over-saturating the strict 256 strict context limit.
slm_llm = CustomSLM()

template = """The following is a conversation with an AI assistant.
{history}
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Keeping window memory stateful globally logic. If multi-user, we would instantiate inside endpoint using a db store, 
# but for a single local user, a global conversational chain works perfectly.
conversation = ConversationChain(
    llm=slm_llm, 
    memory=ConversationBufferWindowMemory(k=3),
    prompt=prompt,
    verbose=True
)

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def get_root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat")
def chat_completion(req: ChatRequest):
    # Langchain seamlessly maps the string directly updating the conversation buffer entirely abstracting string parsing
    response_text = conversation.predict(input=req.query)
    return {"response": response_text}
