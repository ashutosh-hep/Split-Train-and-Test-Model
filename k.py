import pandas as pd

df = pd.read_csv("E:\ww\\dataset.csv")
df.head(100)

X = df[["PRINCIPAL_BUSINESS_ACTIVITY","AUTHORIZED_CAPITAL"]]
y = df[["CORPORATE_IDENTIFICATION_NUMBER","PAIDUP_CAPITAL"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')


X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)
clf = LinearRegression().fit(X_train, y_train)

clf.predict(X_test)

clf.score(X_test, y_test)
