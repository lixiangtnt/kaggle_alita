from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import auc
#from sklearn.metrics import mean_squared_error
#from sklearn import linear_model
import data_utils
import pandas as pd
from sklearn.model_selection import GridSearchCV

train_X, train_Y, test_X = data_utils.load_data()
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)

my_model = XGBClassifier(n_jobs=-1)
my_model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=[(X_valid, Y_valid)], verbose=True)
predictions = my_model.predict_proba(X_valid)[:,0]
from sklearn.metrics import auc, roc_curve
submission = pd.DataFrame(data={"ground_truth": Y_valid, "result": predictions})
fpr, tpr, thresholds = roc_curve(submission['ground_truth'], submission['result'], pos_label=1)
# result_df = result_df.sort_values(by='ground_truth')
valid_auc = auc(fpr, tpr)
print("_auc_{}".format(valid_auc))

# param_dist = {"max_depth": [10,30,50],
#               "min_child_weight" : [1,3,6],
#               "n_estimators": [200],
#               "learning_rate": [0.05, 0.1,0.16],}
#
# grid_search = GridSearchCV(my_model, param_grid=param_dist, cv = 3,
#                                    verbose=10, n_jobs=1)
# grid_search.fit(X_train, Y_train)
#
# predictions = grid_search.predict_proba(X_valid)[:,0]
# result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
# result_df = result_df.sort_values(by='ground_truth')