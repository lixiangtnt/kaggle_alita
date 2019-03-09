from sklearn.model_selection import train_test_split
from sklearn import linear_model
import data_utils
import pandas as pd


train_X, train_Y, test_X = data_utils.load_data()
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, Y_train)
result = clf.predict(X_valid)
result_df = pd.DataFrame(data={"result": result, "ground_truth": Y_valid})
result_df.to_csv("svm_baseline.csv", sep="\t", index=False)