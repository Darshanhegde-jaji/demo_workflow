import importlib
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langgraph.types import Command


client = MongoClient("mongodb://localhost:23017")
mongo_checkpointer = MongoDBSaver(client=client,
                                  db_name="langgraph_memory",
                                  collection_name="checkpoints")

def load_graph(workflow_id: str):
    module_name = workflow_id
    module = importlib.import_module(module_name)
    return getattr(module, "builder") 

app = FastAPI()
builder = load_graph("youtube")
config = {"configurable": {"thread_id": "7000"}}
graph = builder.compile(checkpointer=mongo_checkpointer) 

@app.post("/run")
async def run(input: dict):
    try:
        result = graph.invoke(input, config=config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/resume/")
async def resume(input: str):
    try:
        result = graph.invoke(Command(resume=input), config=config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/state/")
async def state():
    try:
        result = graph.get_state(config) 
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
