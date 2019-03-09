from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import data_utils
import pandas as pd


train_X, train_Y, test_X = data_utils.load_data()
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)
clf = LinearSVC(tol=1e-5)
clf.fit(X_train, Y_train)
result = clf.predict(X_valid)
result_df = pd.DataFrame(data={"result": result, "ground_truth": Y_valid})
result_df.to_csv("linear_svm_baseline.csv", sep="\t", index=False)
acc = result_df[result_df["result"] == Y_valid["ground_truth"]].shape[0]/result_df.shaple[0]
print("acc is {}".format(acc))
