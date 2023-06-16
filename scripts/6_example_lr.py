"""Don't run this function
"""

import numpy as np
from sklearn.linear_model import LogisticRegression as LR
data = np.loadtxt("my_result.csv")
X, y = data[:, :-1], data[:, -1]
alpha = 0.1
lr = LR(penalty="l2", C=1 / alpha)
#
model = lr.fit(X, y)
# 
weights = model.coef_[0]
bias = model.intercept_
print(weights, bias)
#
p_predict = model.predict_proba(X)
# 
y_predict = model.predict(X)
