from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from quillan import QuillanSOTA
import os
from typing import List, Optional

app = FastAPI(title="Quillan API", description="API for Quillan v5.3.1 SOTA Model")

# CORS - allow browser frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GenerateRequest(BaseModel):
    prompt_ids: List[int]
    max_new_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9

class GenerateResponse(BaseModel):
    generated_ids: List[int]

class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "standard"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

class ChatResponse(BaseModel):
    response: str
    thinking: Optional[List[str]] = []
    mode: str
    model_loaded: bool

@app.on_event("startup")
async def load_model_event():
    global model
    checkpoint_path = "checkpoints/quillan_final.pt"
    print(f"Loading model from {checkpoint_path} on {device}...")
    
    model = QuillanSOTA(
        vocab_size=50257,
        dim=512,
        num_mini_moes=32,
        num_experts_per_mini=8,
        num_micros_per_mini=325,
        num_layers=6,
        num_heads=8,
        max_seq_len=2048,
        diffusion_steps=10,
        use_bitnet=True,
        dropout=0.1
    )
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_tensor = torch.tensor([request.prompt_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
        
        return GenerateResponse(generated_ids=output_ids[0].tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Text-based chat endpoint for the frontend interface."""
    thinking_steps = []
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Training may still be in progress."
        )
    
    try:
        thinking_steps.append("Tokenizing input query...")
        # Simple char-level tokenization matching local_inference.py
        tokens = [min(ord(c), 999) for c in request.message]
        max_len = 50
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        
        text_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Dummy multimodal inputs (text-only inference)
        img = torch.randn(1, 3, 256, 256).to(device)
        aud = torch.randn(1, 1, 2048).to(device)
        vid = torch.randn(1, 3, 8, 32, 32).to(device)
        
        thinking_steps.append("Running forward pass through council network...")
        
        with torch.no_grad():
            outputs = model(text_tensor, img, aud, vid)
        
        response_text = ""
        if 'text' in outputs:
            logits = outputs['text']
            predicted_tokens = torch.argmax(logits, dim=-1)
            response_chars = []
            for token in predicted_tokens[0]:
                token_val = token.item()
                if token_val == 0:
                    break
                elif 1 <= token_val <= 999:
                    char_code = min(max(token_val, 32), 126)
                    response_chars.append(chr(char_code))
            response_text = ''.join(response_chars).strip()
        
        thinking_steps.append("Synthesis complete.")
        
        if not response_text:
            response_text = "[Model generated empty response — training may still be in progress]"
        
        return ChatResponse(
            response=response_text,
            thinking=thinking_steps,
            mode=request.mode or "standard",
            model_loaded=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
