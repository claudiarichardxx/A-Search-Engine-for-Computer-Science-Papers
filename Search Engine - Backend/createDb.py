import io
import requests
import pandas as pd, json_normalize
import re
from transformers import BertModel, BertTokenizerFast
from utils.setupFunctions import getStopwords
import torch
from datetime import datetime
from elasticsearch import Elasticsearch
from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from sklearn.linear_model import LogisticRegression

def getHtml(doc_num, mid_string):
  doc_id = str(0) * (5 - len(str(doc_num))) + str(doc_num)
  url = "http://arxiv.org/abs/"+ mid_string+"."+ doc_id
  response=requests.get(url)
  html = response.text
  return html

def getTitle(html):
  return html[html.index('<title>') + 20:html.index('</title>')]

def getAbstract(text):
    start_ind = text.index('"citation_abstract" content=') + 29
    from_abs = text[start_ind:]
    next_new_line = [m.start() for m in re.finditer(r'" />',from_abs)][0] + start_ind
    abstract = text[start_ind:next_new_line]
    abstact = abstract.replace('-','')
    abstact = abstract.replace('\n',' ')
    return abstact

def getAuthors(html):
  auts = html[html.index('citation_author')+ 16:html.index('"citation_date" content="')]
  auts = auts.replace('"citation_author"', '')
  authors = []
  quote_positions = [m.start() for m in re.finditer(r'"', auts)]
  for i in range(0,len(quote_positions), 2):
    name = auts[quote_positions[i]+1: quote_positions[i+1]].replace(' ', '')
    authors.append(name.split(','))
  return authors

def getDate(text):
    start_ind = text.index('"citation_date" content="') + 25
    from_abs = text[start_ind:]
    next_new_line = [m.start() for m in re.finditer(r'" />',from_abs)][0] + start_ind
    date = text[start_ind:next_new_line]
    year = int(date[:4])
    month = int(date[5:7])
    return [date, year, month]

def getYear(arr):
  return arr[1]

def getMonth(arr):
  return arr[2]

def getLink(doc_num):
  doc_id = str(0) * (5 - len(str(doc_num))) + str(doc_num)
  url = "http://arxiv.org/pdf/2301."+ doc_id + ".pdf"
  return url

def buildArchive(archive, num_of_docs, starting_num, mid_string):
  doc_num = starting_num
  lst = []
  while(doc_num<=num_of_docs):
      try:
        html = getHtml(doc_num, mid_string)
      except:
        return archive
      row = {}
      row['DocId'] = doc_num
      try:
        row['Title']= getTitle(html)
      except:
        row['Title'] = ''
      try:
        row['Date']= getDate(html)
        row['Month'] = row['Date'][2]
        row['Year'] = row['Date'][1]

      except:
        row['Date'] = ['',0,0]
      try:
        row['Authors']= getAuthors(html)
      except:
        row['Authors'] = []
      try:
        row['Abstract'] = getAbstract(html)
      except:
        row['Abstract'] = ''
      doc_id = str(0) * (5 - len(str(doc_num))) + str(doc_num)
      row['Link'] = "http://arxiv.org/pdf/"+ mid_string + "." + doc_id + ".pdf"
      row['Cornell_Index'] = mid_string
      doc_num = doc_num + 1
      lst.append(row)
      #archive.loc[len(archive)] = row
      #archive = archive.append(row, ignore_index = True)
  archive = pd.DataFrame(lst)  
  return archive

def preprocess(text):
  return_str = []
  for word in text.split(' '):
    if (word.isalnum()):
      if(word.lower() not in stopwords):
        return_str.append(word)
  return return_str

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def getEmbeddings(text, model, tokenizer):    
    #Tokenize sentences
    #all-mpnet-base-v2_model = BertModel.from_pretrained("Models/")
    #all-mpnet-base-v2_tokenizer = BertTokenizerFast.from_pretrained("Models/Tokenizer")

    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt',is_split_into_words=True)

    #Compute token embeddings
    with torch.no_grad():
      model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embeddings

def deleteIndices():
    for i in range(0,200):
        client.indices.delete(index = i)

def createIndices():
    for i in range(0,200):
        client.indices.create(index = i)
        client.indices.put_mapping(index = i,  body={
        "properties": {
            "Title": {
                "type":  "search_as_you_type"
            },
            "Date": {
                "type":  "date"
            },
            "Authors": {
                "type":  "text"
            },
            "Abstract": {
                "type":  "completion"
            },
            "Embeddings": {
                "type":  "float"
            },
            "Month": {
                "type":  "integer"
            },
            "Year": {
                "type":  "long"
            },
            "Link": {
                "type":  "text"
            },
            "TandA": {
                "type":  "keyword"
            },
            "Keywords_new": {
                "type":  "text"
            },
            "DocKeywords": {
                "type":  "text"
            }}})
        
def index_data(corpus):
  for i, row in (corpus.iterrows()):
      client.index(
      index = row['cluster_id'],
      id = row['DocId'],
      document = {
                  "Title": row['Title'],
                  "Date": row["Date"],
                  "Authors": row["Authors"],
                  "Abstract": row["Abstract"],
                  "Embeddings": row["Embeddings"],
                  "Month": row["Month"],
                  "Year": row["Year"],
                  "Link": row["Link"],
                  "TandA": row["TandA"],
                  "Keywords": row["Keywords_new"],
                  "DocumentKeywords": row["DocKeywords"]
      }
  )

archive = pd.DataFrame()
#buildArchive(archive, num_of_docs, starting_num, mid_string)
print('starting to build archive...')
archive = buildArchive(archive, 5000, 1, '2304')
mini_lm_model = BertModel.from_pretrained("Models/")
mini_lm_tokenizer = BertTokenizerFast.from_pretrained("Models/Tokenizer")
stopwords = getStopwords()
archive['Title_processed'] = archive['Title'].apply(preprocess)
archive['Abstract_processed'] = archive['Abstract'].apply(preprocess)
archive['TandA'] = archive['Title_processed'] + ' ' + archive['Abstract_processed']
archive['Embeddings'] = torch.dtype
#corpus['Month'] = corpus['Date'].apply(getMonth)
print('starting to get embeddings...')
for i in range(0, len(archive)):
  archive.at[i, 'Embeddings'] = getEmbeddings(archive.iloc[i]['TandA'],mini_lm_model, mini_lm_tokenizer)

print('embeddings done!')
print('starting to create indices...')
X = np.array(archive['Embeddings'].tolist())
clust = KMeans(init = 'k-means++', n_clusters = 200, n_init=10)
clust.fit(X)
y = clust.predict(X)
archive['cluster_id'] = y

print('indices created through clustering!')

print('getting keywords...')
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(2,3))
#text_corpus = archive['TandA']
# X = tfidf.fit_transform(text_corpus)
some = archive['TandA']
responses = tfidf.transform(some)
feature_names = np.array(tfidf.get_feature_names_out())
def get_top_tf_idf_words(response, top_n=50):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]]
results = [get_top_tf_idf_words(response,200) for response in responses]
res = []
for doc_result in results:
    #monograms = [i for i in doc_result if len(i.split(' '))==1]
    bigrams = [i for i in doc_result if len(i.split(' '))==2]
    trigrams = [i for i in doc_result if len(i.split(' '))==3]
    delete1 = [e for e in bigrams for i in trigrams if e in i]

    #delete2 = [e for e in monograms for i in bigrams if e in i]

    #delete = delete1 + delete2
    res.append([item for item in doc_result if item not in delete1])

archive['Keywords_new'] = res
dataset  = pd.read_excel('ClassifierResources/Train.xlsx')
X = dataset['Phrases']
y = dataset['comp_word']
# train test split (66% train - 33% test)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
# print('Training Data :', X_train.shape)
# print('Testing Data : ', X_test.shape)
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
lr = LogisticRegression()
lr.fit(X_train_cv, y_train)

# transform X_test using CV
X_test_cv = cv.transform(X_test)

# generate predictions
predictions = lr.predict(X_test_cv)

def getBetterKeywords(words):
    if len(words) == 0:
        return []
    else:
        transformed = cv.transform(words)
        predictions = lr.predict(transformed)
        ind = np.where(np.array(predictions) == 1)[0]
        keywords = np.take(words, ind)
        return keywords
    
archive['Keywords_new'] = archive['keywords'].apply(getBetterKeywords)
def limitWords(words):
  return words[:5]
archive['DocKeywords'] = archive['Keywords_new'].apply(limitWords)



import itertools
import collections

def flatten(x):
  merged = list(itertools.chain.from_iterable(x))
  return merged


print('creating keywords dataframe for manual labeling...')
keywords = pd.DataFrame()
for i in range(0,200):
  row = {}
  row['cluster_id'] = i
  cur = archive[archive['cluster_id']==i].Keywords_new
  crs = flatten(cur)
  #print(collections.Counter(crs))
  dicti = dict(collections.Counter(crs).most_common(30))
  #row['keywords'] = ','.join(list(dicti.keys()))
  row['keywords'] = dicti
  print(row)
  keywords = keywords.append(row, ignore_index = True)

keywords.to_excel('Resources/cluster_df.xlsx', index=False)
print('keywords dataframe created!')
print('Label clusters in Resources/cluster_df.xlsx using column name "cluster_label"')

print('pushing to elasticsearch on http://localhost:9200')
client = Elasticsearch(hosts="http://localhost:9200",verify_certs=False)
print('indexing...')
index_data(archive)
print('all done!')
