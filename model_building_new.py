import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pickle
import xgboost as xgb
data = pd.read_csv('complete_data.csv')
data.drop('Unnamed: 0', axis=1, inplace = True)
data['wage_class'].replace({' <=50K.': ' <=50K',' >50K.':' >50K' }, inplace = True)
# taking refrence from model_building.ipynb file
cat_features = ['relationship', 'marital_status', 'education', 'occupation', 'sex','wage_class']
numerical_feature_list  =['age'	,'capital_gain','hours_per_week']

final_feature_list   = numerical_feature_list+cat_features
final_data = data[final_feature_list]
for feature in cat_features:
  final_data[feature] = final_data[feature].astype('category').cat.codes

x = final_data[numerical_feature_list].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_scaled_df = pd.DataFrame(x_scaled, columns = numerical_feature_list )
data_2 = pd.concat([x_scaled_df,final_data[cat_features]], axis=1)
X = data_2.drop('wage_class',axis=1)
y = data_2['wage_class']
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size = 0.3)
# Init classifier
xgb_cl = xgb.XGBClassifier()

# Fit
xgb_cl.fit(X_train, y_train)

# Predict
preds = xgb_cl.predict(X_test)

print(# accuracy 
accuracy_score(y_test, preds))

file_name = "xgb_classifier_new.pkl"
pickle.dump(xgb_cl, open(file_name, "wb"))
