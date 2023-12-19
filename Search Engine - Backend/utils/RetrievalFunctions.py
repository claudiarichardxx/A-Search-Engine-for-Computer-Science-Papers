#from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi import Depends
from typing_extensions import Annotated
import torch
import config
import numpy as np
from elasticsearch import Elasticsearch
from pandas import json_normalize
import re
import requests
#import logging
#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename=r'C:\Users\LENOVO\Documents\Repos\SearchEngine\app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
#logging.warning('This will get logged to a file')

def get_settings():
    return config.Settings()

def streamlinedProcess(query, settings):
    query_embedding = getQueryEmbedding(query, settings)
    scores = getSimilarityScores(query_embedding, settings.corpus)
    to_return = mergeSort(scores)
    return_df = createJson(to_return, settings.corpus)
    return return_df

def preprocess(text, settings):
  return [term for term in text.split() if term.lower() not in settings.stopwords] 


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def getEmbeddings(text, settings):    
    #Tokenize sentences
    #all_mpnet_base_model = BertModel.from_pretrained("Models/")
    #all_mpnet_base_tokenizer = BertTokenizerFast.from_pretrained("Models/Tokenizer")

    encoded_input = settings.all_mpnet_base_tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt',is_split_into_words=True)

    #Compute token embeddings
    with torch.no_grad():
      model_output = settings.all_mpnet_base_model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embeddings.numpy()


def getCosineSimilarityOld(text1, text2):
  text1_embedding = getEmbeddings(preprocess(text1))
  text2_embedding = getEmbeddings(preprocess(text2))
  return cosine_similarity(text1_embedding, text2_embedding)

def getCosineSimilarity(query, abstract):
  return cosine_similarity(query, abstract)

def mergeSort(arr):
  
    if len(arr) > 1:
 
         # Finding the mid of the array
        mid = len(arr)//2
 
        # Dividing the array elements
        L = arr[:mid]
 
        # Into 2 halves
        R = arr[mid:]
 
        # Sorting the first half
        mergeSort(L)
 
        # Sorting the second half
        mergeSort(R)
 
        i = j = k = 0
 
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i][1] >= R[j][1]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
 
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
 
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
 
    return arr

def getQueryEmbedding(query, settings):

  return getEmbeddings(preprocess(query,settings), settings)


def getSimilarityScores(query_embedding, corpus):
  scores = []
  for doc_id in range(0, len(corpus)):
    score = getCosineSimilarity(query_embedding, corpus.at[doc_id, 'Embeddings'])
    if (score >0):
      scores.append([corpus.at[doc_id, 'DocId'], score])
  return scores

def getSimilarityScores_2(query_embedding, corpus, id_col):
  scores = []
  for doc_id in range(0, len(corpus)):
    score = getCosineSimilarity(query_embedding, corpus.at[doc_id, 'Embeddings'])
    if (score >0):
      scores.append([corpus.at[doc_id, id_col], score])
  return scores

def createJson(scores, corpus):
  scores_df = pd.DataFrame(scores, columns = ['DocId', 'Score'])
  scores_df['DocId'] = scores_df['DocId'].astype(int)
  return_df = pd.merge(scores_df, corpus[['DocId','Title','Abstract','Authors', 'Year','Month','Link', 'cluster_id', 'DocKeywords']], left_on=['DocId'], right_on=['DocId'], how = 'left')
  return_df.rename(columns = {'DocId':'documentId', 'cluster_id' :'clusterId', 'Title': 'documentTitle', 'Keywords':'keywords', 'Link': 'link', 'Year':'year'}, inplace = True)
  return_df['Authors'] = return_df['Authors'].apply(formatAuthors)
  return_list = return_df.to_dict(orient= 'records')
  return return_list, return_df

def formatAuthors(aut):
    #aut = [['Li', 'Tianmu'], ['Li', 'Shurui'], ['Gupta', 'Puneet']]
    #bad = ['citation_doi']
    #aut_st = str([' '.join(i) for i in aut if i not in bad])
    aut_st = str(aut)
    aut_st = aut_st.replace('[','')
    aut_st = aut_st.replace(']','')
    aut_st = aut_st.replace("'",'')
    aut_st = aut_st.replace('citation_doi', '')
    aut_st = re.sub('\d', '', aut_st)
    #aut = [i for i in aut_st if (i.isalpha()==True) or (i == ',')]
    return aut_st


def createJsonSyntactic(return_df):
  return_df['DocId'] = return_df['DocId'].astype(int)
  return_df.rename(columns = {'DocId':'documentId', 'cluster_id' :'clusterId', 'Title': 'documentTitle', 'Keywords':'keywords', 'Link': 'link', 'Year':'year'}, inplace = True)
  return_df['Authors'] = return_df['Authors'].apply(formatAuthors)
  return_list = return_df.to_dict(orient= 'records')
  return return_list, return_df


def docSearchLogic(client, cluster_scores, query_embedding, max_results = 200):
      do = True
      i = -1
      sorted = []
      scores = []
      while(do == True and len(sorted)<=max_results):
        i = i+1
        clust_num = cluster_scores[i][0]
        #df = corpus[corpus['cluster_id']==clust_num].reset_index()
        df = getdata(clust_num, client)
        score_list = getSimilarityScores_2(query_embedding, df, 'DocId')
        scores.extend(score_list)
        sorted = mergeSort(scores)
        do = cluster_scores[i+1][1] > sorted[len(sorted)-1][1]
      return sorted

def createClusterJson(return_df, cluster_df):
  cluster_info = pd.DataFrame(return_df.groupby('clusterId')['documentId'].agg(num_of_docs = 'count', doc_list = lambda x: list(x))).reset_index()
  cluster_info.sort_values(by='num_of_docs', ascending=False, inplace = True)
  cluster_label = []
  cluster_info['clusterId'] = cluster_info['clusterId'].astype(int)
  for clusters in cluster_info.clusterId:
    cluster_label.append(cluster_df.cluster_label[clusters])
  cluster_info['cluster_label'] = cluster_label
  cluster_info.rename(columns = {'cluster_id' : 'clusterId', 'cluster_label': 'clusterName', 'doc_list': 'documentList'}, inplace = True)
  return_clusters = cluster_info.to_dict(orient= 'records')
  return return_clusters

def getdata(clusterId, client):       
       res = client.search(index=clusterId, size= 1000, body={"query": {"match_all": {}}})
       df = json_normalize(res['hits']['hits'])
       df.rename(columns = {'_index':'cluster_label', '_id':'DocId','_source.Title' : 'Title', '_source.Date' : 'Date',
              '_source.Authors': 'Authors', '_source.Abstract': 'Abstract', '_source.Embeddings': 'Embeddings',
              '_source.Month' : 'Month', '_source.Year' : 'Year', '_source.Link' :'Link', '_source.TandA':'TandA',
              '_source.Keywords':'Keywords_new', '_source.DocumentKeywords' : 'DocKeywords'}, inplace = True)

       return df
   
def getDataSyntactic(client, query):       
       res = client.search(size= 200, body={"query": {"bool": {"must": {"match": {      
                                                                      "TandA": query
                                                                    }}}
                                                                }})
       df = json_normalize(res['hits']['hits'])
       df.rename(columns = {'_index':'cluster_id', '_id':'documentId','_source.Title' : 'Title', '_source.Date' : 'Date',
              '_source.Authors': 'Authors', '_source.Abstract': 'Abstract', '_source.Embeddings': 'Embeddings',
              '_source.Month' : 'Month', '_source.Year' : 'Year', '_source.Link' :'Link', '_source.TandA':'TandA',
              '_source.Keywords':'Keywords_new', '_source.DocumentKeywords' : 'DocKeywords'}, inplace = True)

       return df

def getDataSyntactic_2(client, query):       
       resp = requests.post('http://localhost:9200/_all/_search?size=200', json={
                                  "query": {"match": {      
                                        "Title": query
                                      }
                                    }}
                              )
       res = resp.json()
       df = json_normalize(res['hits']['hits'])
       df.rename(columns = {'_index':'cluster_id', '_id':'DocId','_source.Title' : 'Title', '_source.Date' : 'Date',
              '_source.Authors': 'Authors', '_source.Abstract': 'Abstract', '_source.Embeddings': 'Embeddings',
              '_source.Month' : 'Month', '_source.Year' : 'Year', '_source.Link' :'Link', '_source.TandA':'TandA',
              '_source.Keywords':'Keywords_new', '_source.DocumentKeywords' : 'DocKeywords'}, inplace = True)

       return df
