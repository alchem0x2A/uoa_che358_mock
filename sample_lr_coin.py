import numpy as np
from sklearn.linear_model \
    import LogisticRegression as LR
data = np.loadtxt("my_result.csv")
x, y = data[:, :-1], data[:, -1]
alpha = 0.1 # penalty strength
lr = LR(penalty="l2", C=1 / alpha)
# 1. create a model
model = lr.fit(x, y)

# 2. get weight and bias
weights = model.coef_[0]
bias = model.intercept_

# 3. probability
p_predict = model.predict_proba(x)

# 4. predict class
y_predict = model.predict(x)

# 5. accuracy (whole dataset)
acc = model.score(x, y)
