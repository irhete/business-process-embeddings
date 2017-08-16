from sklearn.base import TransformerMixin
import pandas as pd
from time import time
import gensim

class SkipGramActivityResourcePairEmbedding(TransformerMixin):
    
    def __init__(self, case_id_col, activity_col, resource_col, timestamp_col):
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.timestamp_col = timestamp_col
        
        self.fit_time = 0
        self.transform_time = 0
        
        self.wv_size = 15
        
        self.model = None
        
    
    def fit(self, X, y=None):
        start = time()
        act_res_sentences = ActivityResourcePairSentences(X, self.activity_col, self.resource_col)
        self.model = gensim.models.Word2Vec(act_res_sentences, size=self.wv_size, window=1, min_count=1, workers=1)
        self.fit_time = time() - start
        return self
    
    
    def transform(self, X, y=None):
        start = time()
        
        #dt_last = X.groupby(self.case_id_col).last()
        
        # transform numeric cols
        dt_transformed = X[self.resource_col].apply(self._get_vector)
        
        self.transform_time = time() - start
        return dt_transformed
    
    def _get_vector(self, value):
        if value in self.model.wv.vocab:
            return pd.Series(self.model[value])
        else:
            return(pd.Series([0] * self.wv_size))
            
            
class ActivityResourcePairSentences(object):
    def __init__(self, data, activity_col, resource_col):
        self.data = data
        self.activity_col = activity_col
        self.resource_col = resource_col
 
    def __iter__(self):
        for _, row in self.data.iterrows():
            yield [row[self.activity_col], row[self.resource_col]]