from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from .RetrievalFunctions import *



def streamlinedProcess(query, settings):
    query_embedding = getQueryEmbedding(query, settings)
    print(query_embedding)
    scores = getSimilarityScores(query_embedding, settings)
    to_return = mergeSort(scores)
    return_df = createJson(to_return, settings)[:10]
    return return_df