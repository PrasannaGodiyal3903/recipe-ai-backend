from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Recipe Generator API")

# Allow frontend (Lovable or localhost) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once during startup
print("🔹 Loading model... (this may take a while)")
generator = pipeline("text-generation", model="Rivals167/recipe-model")
print("✅ Model loaded successfully!")

# Define input schema
class Input(BaseModel):
    prompt: str

# Root route (for health check)
@app.get("/")
def root():
    return {"message": "✅ Recipe Generator API is live! Use POST /predict"}

# Prediction route
@app.post("/predict")
def predict(data: Input):
    print(f"Received prompt: {data.prompt}")
    result = generator(data.prompt, max_length=150, do_sample=True)
    output_text = result[0]["generated_text"]
    print(f"Generated output: {output_text[:100]}...")  # log partial output
    return {"output": output_text}

# Run the server (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
