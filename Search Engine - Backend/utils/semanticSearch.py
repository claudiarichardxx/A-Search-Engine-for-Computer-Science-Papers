from .RetrievalFunctions import *



def streamlinedSearch(query, settings):
    #query = 'machine learning'
    cluster_predict = settings.clustering_model.predict(settings.tfidf_model.transform([query]))
    
    query_embedding = getQueryEmbedding(query, settings)
    search_df = settings.corpus[settings.corpus['cluster_id'] == cluster_predict[0]].reset_index()
    scores = getSimilarityScores(query_embedding, search_df)
    to_return = mergeSort(scores)
    return_df = createJson(to_return, search_df)
    return return_df    


def streamlinedSearchEmbedding(query, settings):
    #query = 'machine learning'
    query_embedding = getQueryEmbedding(query, settings)
    cluster_scores = mergeSort(getSimilarityScores_2(query_embedding, settings.cluster_df, 'cluster_id'))
    sorted = docSearchLogic(settings.client, cluster_scores, query_embedding, 200)
    doc_json, doc_df = createJson(sorted, settings.corpus)
    cluster_json = createClusterJson(doc_df, settings.cluster_df)
    return {'docss': doc_json, 'clusterss': cluster_json}

def streamlinedElasticSearch(query, settings):
    #query = 'machine learning'
    query_embedding = getQueryEmbedding(query, settings)
    cluster_scores = mergeSort(getSimilarityScores_2(query_embedding, settings.cluster_df, 'cluster_id'))
    sorted = docSearchLogic(settings.client, cluster_scores, query_embedding, 200)
    doc_json, doc_df = createJson(sorted, settings.corpus)
    cluster_json = createClusterJson(doc_df, settings.cluster_df)
    return {'docss': doc_json, 'clusterss': cluster_json}