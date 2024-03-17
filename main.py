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
from network import generate_network
from link_pred import predict_links

def main():
    '''
    Driver function
    '''
    # constants
    queries = [
        'mathematical physics', 'schrodinger operator', 'scattering', 'heat kernel estimates', 'inverse square potential'
    ]

    WINDOW = 1 # Node2Vec fit window
    MIN_COUNT = 1 # Node2Vec min. count
    BATCH_WORDS = 4 # Node2Vec batch words
    
    # fetch data from arXiv
    research_df = search_arxiv(
        queries = queries,
        max_results = 1000
    )
    print(research_df.shape)
    all_tp = research_df.explode('all_topics').copy()
    
    # create network
    tp_nx = generate_network(
        all_tp, 
        node_col = 'article_id', 
        edge_col = 'all_topics'
    )
    print(nx.info(tp_nx))

    # run node2vec
    g_emb = n2v(tp_nx, dimensions=16)

    mdl = g_emb.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )

    # create embeddings dataframe
    emb_df = (
        pd.DataFrame(
            [mdl.wv.get_vector(str(n)) for n in tp_nx.nodes()],
            index = tp_nx.nodes
        )
    )

    print(emb_df.head())
    
    print("Recommended Links to Article: ", predict_links(G = tp_nx, df = emb_df, article_id = 1, N = 10))

    unique_nodes = list(tp_nx.nodes())
    all_possible_edges = [(x,y) for (x,y) in product(unique_nodes, unique_nodes)]

    # generate edge features for all pairs of nodes
    edge_features = [
        (mdl.wv.get_vector(str(i)) + mdl.wv.get_vector(str(j))) for i,j in all_possible_edges
    ]

    # get current edges in the network
    edges = list(tp_nx.edges())

    # create target list, 1 if the pair exists in the network, 0 otherwise
    is_con = [1 if e in edges else 0 for e in all_possible_edges]
    print(sum(is_con))

    # get training and target data
    X = np.array(edge_features)
    y = is_con

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size = 0.3
    )

    # GBC classifier
    clf = GradientBoostingClassifier()

    # train the model
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_true = y_test

    y_train_pred = clf.predict(x_train)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("Testing Accuracy : ", test_acc)
    print("Training Accuracy : ", train_acc)

    print("MCC Score : ", matthews_corrcoef(y_true, y_pred))

    print("Test Confusion Matrix : ")
    print(confusion_matrix(y_pred,y_test))

    print("Test Classification Report : ")
    print(classification_report(y_test, y_pred))

    pred_ft = [(mdl.wv.get_vector(str('42'))+mdl.wv.get_vector(str('210')))]
    print(clf.predict(pred_ft)[0])

    print(clf.predict_proba(pred_ft))
    
if __name__ == '__main__':
    main()