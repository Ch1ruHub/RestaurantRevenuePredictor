"""Portable preprocessing and least-squares prediction pipeline."""
import numpy as np

class RestaurantRevenuePipeline:
    def __init__(self, numeric_columns, categorical_columns):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
    def fit(self, X, y):
        self.means=X[self.numeric_columns].mean().to_dict(); self.scales=X[self.numeric_columns].std(ddof=0).replace(0,1).to_dict()
        self.levels={c:sorted(X[c].astype(str).unique()) for c in self.categorical_columns}; design,self.feature_names=self._design(X)
        self.coefficients=np.linalg.lstsq(design,y.to_numpy(float),rcond=None)[0]; return self
    def _design(self,X):
        arrays,names=[np.ones(len(X))],["Intercept"]
        for c in self.numeric_columns: arrays.append((X[c].to_numpy(float)-self.means[c])/self.scales[c]); names.append(c)
        for c in self.categorical_columns:
            for level in self.levels[c][1:]: arrays.append((X[c].astype(str).to_numpy()==level).astype(float)); names.append(f"{c}={level}")
        return np.column_stack(arrays),names
    def predict(self,X): return self._design(X)[0] @ self.coefficients
