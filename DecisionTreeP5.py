from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np



# 1. adım veri setini tanımla
diabets = load_diabetes()
X = diabets.data # features
y = diabets.target # target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Karar ağacı regresyon modeli
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

mse= mean_squared_error(y_test, y_pred)
print("mse: ", mse)

rmse = np.sqrt(mse)
print("rmse: ", rmse)
