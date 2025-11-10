from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use ["http://localhost", "http://127.0.0.1"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = pipeline("text-generation", model="Rivals167/recipe-model")

class Input(BaseModel):
    prompt: str

@app.post("/predict")
def predict(data: Input):
    result = generator(data.prompt, max_length=150, do_sample=True)
    return {"output": result[0]["generated_text"]}
