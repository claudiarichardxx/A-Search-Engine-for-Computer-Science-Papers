import pandas as pd
import pickle
import bz2file as bz2


def getCorpus():
    return pd.read_json('Data\Archive.json')

def getStopwords():
    #load stopwords
    with open('Resources\stopWords.txt', 'r') as file:
        stopwords = file.read().split('\n')
    return stopwords

def compressed_pickle(title, data):

    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        pickle.dump(data, f)


def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = pickle.load(data)
  return data
