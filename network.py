import networkx as nx
import pandas as pd
import numpy as np
import arxiv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec as n2v
from arxiv_query import search_arxiv

def generate_network(df, node_col = 'article_id', edge_col = 'main_topic'):
    '''
    This function will generate a article to article network given an input DataFrame.
    It will do so by creating an edge_dictionary where each key is going to be a node
    referenced by unique values in node_col and the values will be a list of other nodes
    connected to the key through the edge_col.
    
    params:
        df (DataFrame) : The dataset which holds the node and edge columns
        node_col (String) : The column name associated to the nodes of the network
        edge_col (String) : The column name associated to the edges of the network
        
    returns:
        A networkx graph corresponding to the input dataset
        
    example:
        generate_network(
            research_df,
            node_col = 'article_id',
            edge_col = 'main_topic'
        )
    '''
        
    edge_dct = {}
    
    for i, g in df.groupby(node_col):
        topics = g[edge_col].unique()
        edge_df = df[(df[node_col] != i) & (df[edge_col].isin(topics))]
        edges = list(edge_df[node_col].unique())
        edge_dct[i] = edges

    # create nx network
    g = nx.Graph(edge_dct, create_using = nx.MultiGraph)
    
    return g


if __name__ == "__main__":

    research_df = pd.read_csv('research.csv')
    all_tp = research_df.explode('all_topics').copy()


    tp_nx = generate_network(
        all_tp,
        node_col = 'article_id',
        edge_col = 'all_topics'
    )


    print(nx.info(tp_nx))

    # Applying Node2Vec

    g_emb = n2v(tp_nx, dimensions = 16)

    window  = 1 # Node2vec fit window
    min_count = 1 # Node2vec min count
    batch_words = 4 # Node2vec batch words

    mdl = g_emb.fit(
        window  = window,
        min_count = min_count,
        batch_words = batch_words

    )

    # Create Embedding dataframe

    emb_df = (
        pd.DataFrame(
            [mdl.wv.get_vector(str(n)) for n in tp_nx.nodes()],
            index = tp_nx.nodes
        )
    )

    emb_df.to_csv('emb_export.csv', index = False)

    print(emb_df.head(5))