from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
import uvicorn
import json
from pydantic import BaseModel
from check import spacy_train,bert_train,spacy_get_entities,bert_get_entities
from fastapi.middleware.cors import CORSMiddleware
import os
import yaml
from src.config import logger


# Set up the logger
logs = logger.get_logger(__name__)

with open('config.yaml','r') as f:
    config_data = yaml.safe_load(f)

app = FastAPI()

app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # Replace with your React app's URL
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

app.add_middleware(logger.RouterLoggingMiddleware, logger=logs)

class ModelName(BaseModel):
    model: str

@app.get('/')
async def home_page():
    return "Welcome to the Custom NER Tool"


@app.post('/train')
async def train_ner_model(model:str, file: UploadFile = File(...)):
    if file.content_type != 'application/json':
        logs.error("File is not in the requested format (JSON).")
        raise HTTPException(status_code=400, detail="File is not in the requested format (JSON).")
    contents = await file.read()
    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        logs.error("File content is not valid JSON.")
        raise HTTPException(status_code=400, detail="File content is not valid JSON.")
    
    model_name = model.strip().lower()
    if model_name == 'spacy':
        spacy_output = spacy_train(data)
        logs.info("Spacy model, Training Completed...")
        return {"result": spacy_output}
    elif model_name == 'bert':
        out = bert_train(data)
        logs.info("Bert model, Training Completed...")
        return {"result": out}
    else:
        logs.error("Didn't choose correct model...")
        return {"message": "Choose model between 'spacy' and 'bert'"}

# Function to list available models
def list_available_models():
    try:
        models_folder = 'models'
        available_models = []
        for root, dirs, files in os.walk(models_folder):
            for dir_name in dirs:
                if dir_name in ['bert_base','spacy_base']:
                    available_models.append(os.path.join(root,dir_name))
                else:
                    model_output_path = os.path.join(root, dir_name, 'model_output')
                    if os.path.isdir(model_output_path):
                        available_models.append(model_output_path)
        return available_models
    except Exception as e:
        logs.error(f"An error occurred while checking available models: {e}")
        raise e

@app.get('/available_models')
async def available_models():
    logs.info("Looking available existing models")
    models = list_available_models()
    return {"available_models": models}

@app.get('/entities')
async def get_output(input_text: str, model_url: str = Query(...)):
    if 'spacy' in model_url:
        logs.info("Spacy model selected and getting entities.")
        spacy_entities = spacy_get_entities(input_text, model_url)
        logs.info("Spacy model prediction completed")
        return {"input": input_text, "entities": spacy_entities}
    elif 'bert' in model_url:
        logs.info("Bert model selected and getting entities.")
        bert_output = bert_get_entities(input_text, model_url)
        logs.info("Bert model prediction completed")
        return bert_output
    else:
        logs.error("Invalid model URL. Please provide a valid Spacy or BERT model URL.")
        raise HTTPException(status_code=400, detail="Invalid model URL. Please provide a valid Spacy or BERT model URL.")



if __name__ == "__main__":
    uvicorn.run (
    "main:app",
    port=config_data["uvicorn"]["port"],
    host=config_data["uvicorn"]["host"]
    )
