from .RetrievalFunctions import *



def streamlinedSearchSyntactic(query, settings):
    #query = 'machine learning'
    return_df = getDataSyntactic_2(settings.client, query)
    doc_json, doc_df = createJsonSyntactic(return_df.reset_index())
    cluster_json = createClusterJson(doc_df, settings.cluster_df)
    return {'docss': doc_json, 'clusterss': cluster_json}
