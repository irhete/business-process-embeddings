from sklearn.base import TransformerMixin
import pandas as pd
from time import time
import gensim
from sklearn.preprocessing import MinMaxScaler

class ActivityResourceMatrixTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, activity_col, resource_col, timestamp_col, scale=None):
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.timestamp_col = timestamp_col
        
        self.scale = scale
        
        self.fit_time = 0
        self.transform_time = 0
        
        self.model = None
        
    
    def fit(self, X, y=None):
        start = time()
        self.model = pd.crosstab(X[self.resource_col], X[self.activity_col])
        if self.scale is not None:
            scaler = MinMaxScaler()
            if self.scale == "row":
                tmp = scaler.fit_transform(self.model.T).T
            else:
                tmp = scaler.fit_transform(self.model)
            self.model = pd.DataFrame(tmp, columns=self.model.columns, index=self.model.index)
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
            return pd.Series(self.model[self.model.index == value].iloc[0])
        else:
            return(pd.Series([0] * self.model.shape[1]))
            