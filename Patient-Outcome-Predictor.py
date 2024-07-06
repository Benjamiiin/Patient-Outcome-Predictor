import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
#=====================================================================================================================#
#==================================================DEVELOPMENT========================================================#
#=====================================================================================================================#
# reading the CSV file into a pandas dataframe 
dev_path = "development.csv"
df = pd.read_csv(dev_path)
#---------------------------------------------------------------------------------------#
# here i created a mapping for gender and applied it to the "sex" column in the df
gender_mapping={"male":0, "female":1}
df["sex"]=df["sex"].map(gender_mapping)
#---------------------------------------------------------------------------------------#
# based on the generated correlation heatmaps,
# the information provided for each feature columns,
# their relevance to target column, 
# and finally due to the fact that some of them have many NAN value(more than 3000),
# i decided to drop these columns 
df = df.drop(columns=["Id","edu", "income","race","totcst", "totmcst","meanbp",
                      "wblc","temp","alb","adlp","sod","glucose","urine","adls"])
#---------------------------------------------------------------------------------------#
# here by using 'df.info()' i noticed that some columns have missing values 
# one of them was categorial and the others numerical
# for the categorial i used mode(most frequent value) 
# and for the numerical columns i created a list and looped through each column 
# and fill the missing values with the mean of that column
df["dnr"] = df["dnr"].fillna(df["dnr"].mode().iloc[0])

numerical_columns = ["aps","sps","scoma", "avtisst","prg2m","prg6m","crea","pafi",
                     "bili","ph","surv2m","surv6m","dnrday","charges","bun"]
for col in numerical_columns:
    df[col].fillna(df[col].mean(), inplace=True)
#---------------------------------------------------------------------------------------#
#after dropping 'income' and 'race' there were 4 categorial columns left 
# here i decided to use Onehotencoding from sklearn for two of them
# at first it transforms these columns into a Onehot encoded format 
# by getting the unique feature name of each column out 
# then the newly created columns are added back to the df 
# and the original categorial columns are dropped from the df
# also at the end i casted the new columns to the int64 data type (they had become int32!!)
onehot_enc = OneHotEncoder(sparse_output=False)   
onehot_encoded = onehot_enc.fit_transform(df[["dzgroup","dzclass"]])
onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_enc.get_feature_names_out(["dzgroup", "dzclass"]))
df = pd.concat([df, onehot_df], axis=1)
df[onehot_df.columns] = df[onehot_df.columns].astype("int64")
df.drop(columns=["dzgroup", "dzclass"], inplace=True)

#---------------------------------------------------------------------------------------#
# here num.co represents the number of simultaneous diseases a patient can have
# after examining the relation(corr heatmaps) between this column and the death column 
# i noticed that whenever a patient have more than one diseases it usually results in their death
# as for the line it sets the value to 1 if the value is greater than 1 otherwise it sets it to 0 
df["num.co"] = df["num.co"].apply(lambda x: 1 if x > 1 else 0)
#---------------------------------------------------------------------------------------#
# after examining the i noticed that when the survival estimate 
# whether conducted by a model or the physician is over 0.6 the patient usually tend to survive
# i adjusted it a few time to get the best value
# so i tranformed the four columns and set their value 
# to 1 if the value is less than 0.6 otherwise set it to 0
df["prg2m"] = df["prg2m"].apply(lambda x: 1 if x < 0.6 else 0)
df["prg6m"] = df["prg6m"].apply(lambda x: 1 if x < 0.6 else 0)
df["surv2m"] = df["surv2m"].apply(lambda x: 1 if x < 0.6 else 0)
df["surv6m"] = df["surv6m"].apply(lambda x: 1 if x < 0.6 else 0)
#---------------------------------------------------------------------------------------#
# here for the other 2 categorial columns by using a lambda function similar to before i tranformed the columns
# for the first one i set its value to 1 if a patient has cancer and it has already spread out 0 if the patient dont have cancer
# for the second one i set the value to 1 if the patient has dnr whether its after or before sadm and 0 otherwise
df['ca'] = df['ca'].apply(lambda x: 1 if x in ["yes", "metastatic"] else 0)

df['dnr'] = df['dnr'].apply(lambda x: 1 if x in ["dnr after sadm", "dnr before sadm"] else 0)

#---------------------------------------------------------------------------------------#
# some columns had a wide range of values and in order for my model to make sense of them i decided to normalize them
# so i created a list of the columns and applied a scaler but instead of 'standardscaler' or 'minmaxscaler' 
# i used RobustScaler cause it uses the median and interquartile range for scaling 
# and this makes it more robust to outliers
columns_to_normalize = ["pafi","dnrday", "charges","bun", "sps","avtisst","adlsc"]
scaler = RobustScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

#---------------------------------------------------------------------------------------#
# here i separated features and the target variable from the df
# dropped death column and assigned the remaining to X
# and assigned the death column to y
X = df.drop(columns=["death"])
y = df["death"]

# spliting the data into training and testing sets
x_train1, x_test, y_train1, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
#---------------------------------------------------------------------------------------#
# since i have imbalance classes i decided to use 
# adaptive synthetic sampling to make up for the minority class
# i applied ADASYN to the training data to try to balance the classes 
# by creating synthetic examples of the minority class
ada = ADASYN(random_state=42)
x_train, y_train = ada.fit_resample(x_train1, y_train1)

#=====================================================================================================================#
#==========================================BALANCED RF CLASSIFIER=====================================================#
#=====================================================================================================================#
# here after comparing other classifiers i decided to go with Balanced rf
# first i used Grid search to find the best parameters also i used class_weight=balanced 
# which might seems redundant given that i used balanced rf but to my suprise it gave me a better score                                                 
randomforest = BalancedRandomForestClassifier(n_estimators = 1000, class_weight="balanced", max_depth=30,
                                       max_features="sqrt",min_samples_leaf=1, min_samples_split=2, 
                                       bootstrap=False, sampling_strategy="all",random_state = 42, replacement=True)

# fit the Balanced rf on the training data
randomforest.fit(x_train, y_train)
# predict the target variable death on the test set
y_pred = randomforest.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'F1 score: {f1_score(y_test, y_pred, zero_division=1):.2f}')

#=====================================================================================================================#
#=====================================================================================================================#



#=====================================================================================================================#
#================================================EVALUATION===========================================================#
#=====================================================================================================================#
# here i did the same preprocessing of the data with one difference 
# some columns in development.csv that had no NaN values had NaN values in evaluation.csv and vice versa
# so after using df2.info() i had to change my numerical_columns_eval a bit so that i can fill missing values 

eva_path = "evaluation.csv"
df2 = pd.read_csv(eva_path)
#---------------------------------------------------------------------------------------#
df2["sex"]=df2["sex"].map(gender_mapping)
#---------------------------------------------------------------------------------------#
df2 = df2.drop(columns=["Id","edu", "income","race","totcst", "totmcst","meanbp",
                        "wblc","temp","alb","adlp","sod","glucose","urine","adls"])
#---------------------------------------------------------------------------------------#

df2["dnr"] = df2["dnr"].fillna(df2["dnr"].mode().iloc[0])

numerical_columns_eval = ["charges","avtisst", "dnrday","prg2m","prg6m", "hrt", 
                          "resp", "pafi", "bili", "crea","ph", "bun"] 
for col in numerical_columns_eval:
    df2[col].fillna(df2[col].mean(), inplace=True)
#---------------------------------------------------------------------------------------#

onehot_encoded2 = onehot_enc.fit_transform(df2[["dzgroup","dzclass"]])
onehot_df2 = pd.DataFrame(onehot_encoded2, columns=onehot_enc.get_feature_names_out(["dzgroup", "dzclass"]))
df2 = pd.concat([df2, onehot_df2], axis=1)
df2[onehot_df2.columns] = df2[onehot_df2.columns].astype("int64")
df2.drop(columns=["dzgroup", "dzclass"], inplace=True)
#---------------------------------------------------------------------------------------#
df2["num.co"] = df2["num.co"].apply(lambda x: 1 if x > 1 else 0)
df2["prg2m"] = df2["prg2m"].apply(lambda x: 1 if x < 0.6 else 0)
df2["prg6m"] = df2["prg6m"].apply(lambda x: 1 if x < 0.6 else 0)
df2["surv2m"] = df2["surv2m"].apply(lambda x: 1 if x < 0.6 else 0)
df2["surv6m"] = df2["surv6m"].apply(lambda x: 1 if x < 0.6 else 0)
#---------------------------------------------------------------------------------------#

df2['ca'] = df2['ca'].apply(lambda x: 1 if x in ["yes", "metastatic"] else 0)

df2['dnr'] = df2['dnr'].apply(lambda x: 1 if x in ["dnr after sadm", "dnr before sadm"] else 0)
#---------------------------------------------------------------------------------------#

df2[columns_to_normalize] = scaler.fit_transform(df2[columns_to_normalize])
#---------------------------------------------------------------------------------------#

df2_Predicted = randomforest.predict(df2)
#---------------------------------------------------------------------------------------#
sub = pd.DataFrame({
    "Id": range(len(df2_Predicted)), # generating ids from 0 to the length of df2_Predicted
    "Predicted": df2_Predicted # store predictions from df2_Predicted in the Predicted column
})
# saving sub to a file named 308324".csv without including the index column
sub.to_csv("308324.csv", index=False)
#=====================================================================================================================#
#=====================================================================================================================#
#=====================================================================================================================#





#=====================================================================================================================#
#=====================================================================================================================#
#=====================================================================================================================#
# USED FOR DATA ANALYSIS & MANIPULATION CURRENTLY COMMENTED OUT BECAUSE THEY ARE NOT NECESSARY FOR RUNNING THE PROJECT#
#=====================================================================================================================#
#=====================================================================================================================#

# uniquevalues = df["dzgroup"].unique()
# print("unique values are:", uniquevalues)
#=====================================================================================================================#
# (unique, count) = np.unique(df["death"], return_counts=True)
# print(unique, count)
# sns.barplot(x=unique, y=count)
# plt.xlabel("class")
# plt.ylabel("num of samples")
# plt.xticks()
# plt.title("target variable count in dataset")
# plt.show()
#=====================================================================================================================#
# columns_of_interest = ['death', "age", "num.co"] 
# correlation_matrix = df[columns_of_interest].corr()
# plt.figure(figsize=(20, 20))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
# plt.title('Correlation Heatmap')
# plt.show()
#=====================================================================================================================#
# import statsmodels.api as sm
# from patsy.highlevel import dmatrices
# categorical_column = 'dnr'
# target_column = 'death'
# y, X = dmatrices(f'{target_column} ~ C({categorical_column})', df, return_type='dataframe')
# logit_model = sm.Logit(y, X)
# result = logit_model.fit()
# print(result.summary())
#=====================================================================================================================#
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 500, 1000],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# randomforest = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=randomforest, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(x_train, y_train)

# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_
#=====================================================================================================================#
#=====================================================================================================================#