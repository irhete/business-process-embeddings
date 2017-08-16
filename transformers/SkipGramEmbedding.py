from sklearn.base import TransformerMixin
import pandas as pd
from time import time
import gensim

class SkipGramEmbedding(TransformerMixin):
    
    def __init__(self, case_id_col, activity_col, timestamp_col, wv_size=10):
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        
        self.fit_time = 0
        self.transform_time = 0
        
        self.wv_size = wv_size
        
        self.model = None
        
    
    def fit(self, X, y=None):
        start = time()
        sentences = MySentences(X, self.case_id_col, self.activity_col, self.timestamp_col)
        self.model = gensim.models.Word2Vec(sentences, size=self.wv_size, window=1, min_count=1, workers=4)
        self.fit_time = time() - start
        return self
    
    
    def transform(self, X, y=None):
        start = time()
        
        #dt_last = X.groupby(self.case_id_col).last()
        
        # transform numeric cols
        #dt_transformed = dt_last[self.activity_col].apply(self._get_activity_vector)
        
        dt_transformed = X[self.activity_col].apply(self._get_activity_vector)
        
        self.transform_time = time() - start
        return dt_transformed
    
    def _get_activity_vector(self, activity):
        if activity in self.model.wv.vocab:
            return pd.Series(self.model[activity])
        else:
            return(pd.Series([0] * self.wv_size))
        
    
    
class MySentences(object):
    def __init__(self, X, case_id_col, activity_col, timestamp_col):
        self.data = X
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
 
    def __iter__(self):
        grouped = self.data.sort_values(self.timestamp_col, ascending=True).groupby(self.case_id_col)
        for _, group in grouped:
            yield group[self.activity_col].tolist()