from sklearn.base import TransformerMixin
import pandas as pd
from time import time
import gensim

class SkipGramActivityResourceUnitedEmbedding(TransformerMixin):
    
    def __init__(self, case_id_col, activity_col, resource_col, timestamp_col):
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.timestamp_col = timestamp_col
        
        self.fit_time = 0
        self.transform_time = 0
        
        self.wv_size = 30
        
        self.model = None
        
    
    def fit(self, X, y=None):
        start = time()
        act_res_united_sentences = ActivityResourceUnitedSentences(X, self.case_id_col, self.timestamp_col, [self.activity_col, self.resource_col])
        self.model = gensim.models.Word2Vec(act_res_united_sentences, size=self.wv_size, window=2, min_count=1, workers=1)
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
        
    
    
class ActivityResourceUnitedSentences(object):
    def __init__(self, data, case_id_col, timestamp_col, cols):
        self.data = data
        self.cols = cols
        self.case_id_col = case_id_col
        self.timestamp_col = timestamp_col
 
    def __iter__(self):
        grouped = self.data.sort_values(self.timestamp_col, ascending=True).groupby(self.case_id_col)
        for _, group in grouped:
            yield group[self.cols].values.flatten().tolist()