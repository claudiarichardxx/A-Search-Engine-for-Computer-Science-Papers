from transformers import BertModel, BertTokenizerFast
from utils.setupFunctions import getCorpus, getStopwords, decompress_pickle
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from elasticsearch import Elasticsearch
import pandas as pd
from typing import ClassVar


class Settings(BaseSettings):
    all_mpnet_base_model:BertModel = BertModel.from_pretrained("Models\BertModel")
    all_mpnet_base_tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained("Models\BertTokenizer")
    corpus:pd.DataFrame = getCorpus()
    stopwords:list = getStopwords()
    rows:int = 200
    cluster_df: pd.DataFrame = pd.read_excel('Resources/cluster_df.xlsx', engine='openpyxl')
    client: ClassVar[Elasticsearch]   = Elasticsearch("http://localhost:9200")
    #clustering_model:KMeans = decompress_pickle('Models\KMeans\cluster_model_final.pbz2')
    #tfidf_model:TfidfVectorizer = decompress_pickle(r'Models\TFIDF\tfidf_model_final.pbz2')
    model_config = SettingsConfigDict(env_file=".env")
