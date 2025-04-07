#Sigmoid Visualization
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid(True)
plt.show()


#Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#Load IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# Filter only two classes
binary_filter = y != 2
X = X[binary_filter]
y = y[binary_filter]


#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#predict probabilities
y_prob = model.predict_proba(X_test)
print("Predicted Probabilities:\n", y_prob[:5])


#Multinomial Logistic Regression
from sklearn.linear_model import LogisticRegression

# Full iris dataset with 3 classes
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

multi_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
multi_model.fit(X_train, y_train)

y_pred = multi_model.predict(X_test)
print(classification_report(y_test, y_pred))


#Ordinal Logistic Regression (via statsmodels)
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Synthetic data
df = pd.DataFrame({
    'x1': np.random.normal(size=100),
    'x2': np.random.normal(size=100),
    'y': np.random.choice(['low', 'medium', 'high'], size=100, p=[0.3, 0.5, 0.2])
})

df['y_cat'] = pd.Categorical(df['y'], categories=['low', 'medium', 'high'], ordered=True)
model = OrderedModel(df['y_cat'], df[['x1', 'x2']], distr='logit')
res = model.fit(method='bfgs')
print(res.summary())


