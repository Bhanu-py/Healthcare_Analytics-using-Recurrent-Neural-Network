import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
# from autokeras import StructuredDataClassifier
# from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import time

# print(tensorflow.__version__)
# print(keras.__version__)
t1 = time.time()

df_train = pd.read_csv(r"E:\Hackathon\Janatahack Healthcare Analytics II\Train_hMYJ020\train.csv")
df_test = pd.read_csv(r"E:\Hackathon\Janatahack Healthcare Analytics II\Test_ND2Q3bm\test.csv")
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)

df_train['Bed Grade'] = df_train['Bed Grade'].fillna(2.0).astype(int)
df_test['Bed Grade'] = df_test['Bed Grade'].fillna(2.0).astype(int)
df_train['City_Code_Patient'] = df_train['City_Code_Patient'].fillna(method='bfill').astype(int)
df_test['City_Code_Patient'] = df_test['City_Code_Patient'].fillna(method='bfill').astype(int)
df_test['Admission_Deposit'] = df_test['Admission_Deposit'].astype(int)
# df_train['City_Code_Patient'] = df_train['City_Code_Patient'].fillna(2.0, method='ffill').astype(int)
# df_train['City_Code_Patient'] = my_imputer.fit_transform(df_train['City_Code_Patient'])

le = LabelEncoder()
df_train['Hospital_type_code'] = le.fit_transform(df_train["Hospital_type_code"])
df_train['Hospital_region_code'] = le.fit_transform(df_train["Hospital_region_code"])
df_train['Department'] = le.fit_transform(df_train["Department"])
df_train['Ward_Type'] = le.fit_transform(df_train["Ward_Type"])
df_train['Ward_Facility_Code'] = le.fit_transform(df_train["Ward_Facility_Code"])
df_train['Type of Admission'] = le.fit_transform(df_train["Type of Admission"])
df_train['Severity of Illness'] = le.fit_transform(df_train["Severity of Illness"])
df_train['Age'] = le.fit_transform(df_train["Age"])
# df_train['Stay'] = le.fit_transform(df_train["Stay"])


df_test['Hospital_type_code'] = le.fit_transform(df_test["Hospital_type_code"])
df_test['Hospital_region_code'] = le.fit_transform(df_test["Hospital_region_code"])
df_test['Department'] = le.fit_transform(df_test["Department"])
df_test['Ward_Type'] = le.fit_transform(df_test["Ward_Type"])
df_test['Ward_Facility_Code'] = le.fit_transform(df_test["Ward_Facility_Code"])
df_test['Type of Admission'] = le.fit_transform(df_test["Type of Admission"])
df_test['Severity of Illness'] = le.fit_transform(df_test["Severity of Illness"])
df_test['Age'] = le.fit_transform(df_test["Age"])
# print(df_train.head(5))

df1 = df_test.copy()
df2 = df_train.copy()

# df1 = pd.get_dummies(df1, columns=["Stay"])
# print(df1.head(5))
df2 = pd.get_dummies(df2, columns=['Stay'], prefix="", prefix_sep="")
# print(df2.head(5))
df2_x = df2.drop(columns=["case_id", "patientid"], axis=1)
df2_y = df2[['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'More than 100 Days']]
# print(df2.columns.tolist())

train_x = df_train.drop(columns=["case_id", "Stay", "patientid"], axis=1)
train_y = df_train["Stay"]


test_x = df_test.drop(columns=["case_id", "patientid"], axis=1)
print(train_y.head(3))
# print(df2_x.head(3))
# print(df2_y.head(3))
# print(df2_y.shape)
# print(df2_x.shape)
t2 = time.time()
print(f"Data Processed --- {t2-t1}sec")

# clf = GaussianNB()	#27.6221225056
# clf = BernoulliNB() #20.03

# clf = MLPClassifier(max_iter=300)   #36.465
# clf = CatBoostClassifier(n_estimators=10000, random_state=2020, eval_metric='Accuracy', learning_rate=0.08, depth=8, bagging_temperature=0.3,)   #39.9109846412024
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf = KNeighborsClassifier(n_neighbors=10)

clf = Sequential()
clf.add(Dense(30, input_dim=26, kernel_initializer="he_uniform", activation="relu"))
clf.add(Dense(11, activation="sigmoid"))
clf.compile(loss="binary_crossentropy", optimizer='adam')

clf.fit(df2_x, df2_y, verbose=0, epochs=100)

# search = StructuredDataClassifier(max_trials=15)
# clf.fit(train_x, train_y)
t3 = time.time()
print(f"Model Created and Trained ---- {(t3-t2)/60}min")

sample = clf.predict(test_x)
# sample = sample.ravel()

stay = {0: "0-10", 1: "11-20", 2: "21-30", 3: "31-40", 4: "41-50", 5: "51-60", 6: "61-70", 7: "71-80", 8: "81-90", 9: "91-100", 10: "More than 100 Days"}

result = np.vectorize(stay.get)(sample)

t4 = time.time()
print(f"Prediction Done! ---- {t4-t3} sec ")

print(sample)
print(result)
print()
print((type(result)))

submit = pd.DataFrame(({"case_id": df_test.case_id, "Stay": result}))

submit.to_csv("submission.csv", index=False)

print(f"submission.csv File created! ---- {(t4-t1)/60}min")





# f = plt.figure(figsize=(19, 17))
# plt.matshow(df_train.corr(), fignum=f.number)
# plt.xticks(range(df_train.shape[1]), df_train.columns, fontsize=11, rotation=90)
# plt.yticks(range(df_train.shape[1]), df_train.columns, fontsize=11)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()
#
# corr = df_train.corr()
# corr.style.background_gradient(cmap='coolwarm')
#
# print(corr)
