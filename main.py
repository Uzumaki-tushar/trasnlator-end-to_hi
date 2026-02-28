from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# 1. Load the specific model and tokenizer manually
# This avoids the "translation task" naming error entirely.
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "English to Hindi Translator (Manual Load)"}

@app.post("/translate")
async def translate_text(request: TextRequest):
    # 2. Convert text to numbers (tokens)
    inputs = tokenizer(request.text, return_tensors="pt")

    # 3. Generate translation
    translated_tokens = model.generate(**inputs)

    # 4. Convert numbers back to Hindi text
    hindi_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return {
        "original_text": request.text,
        "translated_text": hindi_text
    }