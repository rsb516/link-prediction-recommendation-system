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


# We will focus on the following queries
'''
queries = [
    'automl', 'machine learning', 'data', 'physics', 'mathematics', 'recommendation system', 'nlp', 'neural networks'
]
'''

def search_arxiv(queries, max_results = 100):
    '''
    This function will search arxiv associated to a set of queries and store
    the latest 10000 (max_results) associated to that search.
    
    params:
        queries (List -> Str) : A list of strings containing keywords you want
                                to search on Arxiv
        max_results (Int) : The maximum number of results you want to see associated
                            to your search. Default value is 1000, capped at 300000
                            
    returns:
        This function will return a DataFrame holding the following columns associated
        to the queries the user has passed. 
            `title`, `date`, `article_id`, `url`, `main_topic`, `all_topics`
    
    example:
        research_df = search_arxiv(
            queries = ['automl', 'recommender system', 'nlp', 'data science'],
            max_results = 10000
        )
    '''

    #d = pd.DataFrame()
    d = []
    searches  = []

    # making request from the API

    for query in queries:
        search  = arxiv.Search(
            query = query,
            max_results = max_results,
            sort_by = arxiv.SortCriterion.SubmittedDate,
            sort_order = arxiv.SortOrder.Descending
        )
        searches.append(search)

    # Making the search results into a dataframe
    for search in searches:
        for res in search.results():
            data = {
                'title': res.title,
                'date': res.published,
                'article_id': res.entry_id,
                'url': res.pdf_url,
                'main_topic': res.primary_category,
                'all_topics': res.categories,
                'authors': res.authors
    
            }
            d.append(data)
        
    d = pd.DataFrame(d)
    d['year'] = pd.DatetimeIndex(d['date']).year


    #Changing article id from url to integer
    unique_article_ids = d['article_id'].unique()
    article_mapping = {art:idx for idx, art in enumerate(unique_article_ids)}
    d['article_id'] = d['article_id'].map(article_mapping)

    return d


if __name__ == "__main__":   

    research_df = search_arxiv(
        queries = queries,
        max_results = 1000
    )

    # We output a csv file for the request to the API because their is a delay when hitting it, and it doesn't always work
    research_df.to_csv('research.csv', index = False)
    print(research_df.shape)
    print(research_df.head(2))