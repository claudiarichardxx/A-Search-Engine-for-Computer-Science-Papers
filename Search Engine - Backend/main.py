#from utils.DocumentRetrieval import *
from utils.RetrievalFunctions import *
from utils.semanticSearch import *
from utils.syntacticSearch import *
from functools import lru_cache
import config
from fastapi import Depends, FastAPI, Response, Request, Body
from typing_extensions import Annotated
from typing import Any
from fastapi import status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json
#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename=r'C:\Users\LENOVO\Documents\Repos\SearchEngine\app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
#logging.warning('This will get logged to a file')
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5174",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache
def get_settings():
    return config.Settings()

@app.post("/semantic_search")
def hello(settings: Annotated[config.Settings, Depends(get_settings)], payload: Any = Body(None)):
    query = payload['query']

    if query is None:
        return_json = {'Response': 'No Query'}

    else:
        if(payload['type']==1):

            return_json = streamlinedSearchEmbedding(query, settings)
        else:
            return_json = streamlinedSearchSyntactic(query, settings)

    json_str = json.dumps(return_json, indent=2, default=str)
    return Response(content=json_str, media_type="application/json")
    #return JSONResponse(content=jsonable_encoder(return_json), status_code=status.HTTP_201_CREATED)