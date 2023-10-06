import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import clean_data as cd


# ==========Task No.1=============

# Read and review csv file.
nyc = pd.read_csv("data.csv", sep=';')

print(nyc.head())
print(nyc.tail())


# Clean data. Review the data.
nyc_dict = {}
for string in nyc.values:
    data_list = string[0].split(",")
    if cd.y_verify(data_list[0]):
        year = data_list[0]
        year = int(year[:4])
        temperature = float(data_list[1])
        if temperature > 0:
            nyc_dict.update({year: temperature})

nyc_clean = pd.DataFrame(nyc_dict.items(), columns=["Date", "Temperature"])


# Split data in train and test. Create and train model using split data.
X_train, X_test, y_train, y_test = train_test_split(nyc_clean.Date.values.reshape(-1, 1),
                                                    nyc_clean.Temperature.values, random_state=11)

print(nyc_clean.head(), nyc_clean.tail(), sep="\n")

linear_regression = LinearRegression()

linear_regression.fit(X=X_train, y=y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)

predicted = linear_regression.predict(X_test)
expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")


predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)


# Split cleaned data in before 2019 and after 2018 to train and test model.
learning_nyc_dict = {i: e for i, e in nyc_dict.items() if i < 2019}
learning_nyc = pd.DataFrame(learning_nyc_dict.items(), columns=["Year", "Temperature"])

for_check_nyc_dict = {i: e for i, e in nyc_dict.items() if i > 2018}
for_check_nyc = pd.DataFrame(for_check_nyc_dict.items(), columns=["Year", "Temperature"])


# Create the model.
x = np.array(learning_nyc.Year).reshape((-1, 1))
y = np.array(learning_nyc.Temperature)

model_regression = LinearRegression()
model_regression.fit(x, y)

print(f"The temperature in 1890 was{predict(1890)}")
print(f"The temperature in 1890 was {model_regression.coef_ * 1890 + model_regression.intercept_}")

print(f"{linear_regression.score(X_test, y_test): .2%}")
print(f"{model_regression.score(X_test, y_test): .2%}")

# Perform diagram for learning and predicted.
axes = sns.scatterplot(data=nyc_clean, x="Date", y="Temperature",
                       hue="Temperature", palette="winter", legend=False)
axes.set_ylim(10, 70)

x = np.array([min(nyc_clean.Date.values), max(nyc_clean.Date.values)])
y = predict(x)

line = plt.plot(x, y)

plt.show()


# ==========Task No.2=============

# Load data from built-in dataset.
vines = datasets.load_wine()
features = vines.data
target = vines.target

# Split data in train and test.
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=11,
                                                    train_size=0.75, test_size=0.25)

# Create SVC model and train.
svc = SVC(kernel="linear", random_state=0, gamma=1)
svm_classifier = svc.fit(X_train, y_train)

# Test the model.
predicted_svm = svm_classifier.predict(X=X_test)
expected_svm = y_test

# Estimate the model.
print(predicted_svm[:20], expected_svm[:20], sep="\n")
print(f"{svm_classifier.score(X_test, y_test): .2%}", end="\n")


# ==========Task No.3=============

# Create KNN model and train.
knn = KNeighborsClassifier()

pipe = Pipeline([("knn", knn)])

search_space = [{"knn__n_neighbors": range(1, 12)}]

knn_classifier = GridSearchCV(pipe, search_space, cv=5, scoring="accuracy").fit(X_train, y_train)

best_k = knn_classifier.best_estimator_.get_params()['knn__n_neighbors']

knn = KNeighborsClassifier(n_neighbors=best_k)

# Test the model.
predicted_knn = knn_classifier.predict(X=X_test)
expected_knn = y_test

# Estimate the model.
print(predicted_knn[:20], expected_knn[:20], sep="\n")
print(f"{knn_classifier.score(X_test, y_test): .2%}")


# Create NB model and train.
nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

# Test the model.
predicted_nb = nb_classifier.predict(X=X_test)
expected_nb = y_test

# Estimate the model.
print(predicted_nb[:20], expected_nb[:20], sep="\n")
print(f"{nb_classifier.score(X_test, y_test): .2%}")
