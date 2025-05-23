import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from peft import PeftModel
import torch

# Set Streamlit title
st.title("Text Summarizer with Pegasus (LoRA Fine-tuned)")

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = PegasusTokenizer.from_pretrained("pegasus-lora")
    base_model = PegasusForConditionalGeneration.from_pretrained("pegasus-xsum")
    model = PeftModel.from_pretrained(base_model, "pegasus-lora")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Text input
text = st.text_area("Enter text to summarize", height=200)

# Summarize on button click
if st.button("Summarize"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=100, min_length=30, num_beams=4)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text before clicking summarize.")
