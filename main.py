from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from peft import PeftModel
import torch

# Initialize FastAPI
app = FastAPI(title="Summarizer API")

# Load tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained("pegasus-lora")
base_model = PegasusForConditionalGeneration.from_pretrained("pegasus-xsum")
model = PeftModel.from_pretrained(base_model, "pegasus-lora")
model.eval()

# Define request body
class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(request: TextRequest):
    text = request.text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100, min_length=30, num_beams=4)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"summary": summary}
