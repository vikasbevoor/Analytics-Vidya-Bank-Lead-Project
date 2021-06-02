#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv(r"C:\Users\Admin\Downloads\train_set.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


# Dropping the "ID" column as it is not useful in the analysis


# In[8]:


data = data.drop(columns="ID", axis=1)
data.head()


# ## Exploratory Data Analysis

# In[9]:


# Checking the NA values in the dataset


# In[10]:


data.isna().sum()


# #### Only "Credit_Product" has missing values

# In[11]:


percentage =(data["Credit_Product"].isna().sum()/len(data["Credit_Product"])) * 100
print("Missing values in the Credit_Product - ","%.2f" %percentage, "%")


# ### Numerical Variables in the data

# In[12]:


numerical_variables = [feature for feature in data.columns if data[feature].dtype != "O"]


# In[13]:


numerical_variables


# In[14]:


del numerical_variables[3]


# In[15]:


numerical_variables


# In[16]:


data[numerical_variables].head()


# In[17]:


data.nunique()


# In[18]:


data["Is_Lead"].unique()


# ### Univariate Analysis

# In[19]:


# Using histograms to analyze the distributions of the data


# In[20]:


for feature in numerical_variables:
    data[feature].hist(bins=20)
    plt.title("Histogram of "+ feature)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()


# ####  'Age', 'Vintage', 'Avg_Account_Balance' are not normally distributed but they right skewwed

# In[21]:


# Boxplots to identify the outlier values


# In[22]:


for feature in numerical_variables:
    data.boxplot(column = feature)
    plt.title("Boxplot of "+ feature)
    plt.show()


# #### There are lot of outliers in the 'Avg_Account_Balance'

# In[23]:


dataset = data


# In[24]:


# Identifying the outliers using IQR method


# In[25]:


Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1


# In[26]:


# Number of outliers in the variable


# In[27]:


((dataset < (Q1 - 3 * IQR)) | (dataset > (Q3 + 3 * IQR))).sum()


# #### There are many outliers in the "Avg_Account_Balance"

# In[28]:


IQR_data = dataset["Avg_Account_Balance"].quantile(0.75) - dataset["Avg_Account_Balance"].quantile(0.25)
lower_bridge = dataset["Avg_Account_Balance"].quantile(0.25) - (IQR_data * 3)
upper_bridge = dataset["Avg_Account_Balance"].quantile(0.75) + (IQR_data * 3)
print(lower_bridge,'\n',upper_bridge)


# In[29]:


dataset["Is_Lead"].unique()


# In[30]:


dataset.loc[dataset["Avg_Account_Balance"]> upper_bridge, "Avg_Account_Balance"] = upper_bridge


# In[31]:


((dataset < (Q1 - 3 * IQR)) | (dataset > (Q3 + 3 * IQR))).sum()


# ### Bivariate Analysis

# In[32]:


# Using boxplots to understand the relationship between the input numerical variables and output


# In[33]:


for feature in numerical_variables:
    sns.boxplot("Is_Lead", feature, data=dataset, palette="hls")
    plt.title("Boxplot")
    plt.show()


# In[34]:


# Countplot of dependent variable


# In[35]:


sns.countplot("Is_Lead", data= dataset, palette='hls')


# In[36]:


dataset["Is_Lead"].value_counts()


# In[37]:


inter = sum(dataset["Is_Lead"] == 1)/sum(dataset["Is_Lead"]==0) * 100
non_inter = 100 - inter

print("Interested customers", "%.2f" %inter, "%")
print("Not interested customers","%.2f" %non_inter, "%")


# ### Categorical Variables

# In[38]:


categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == "O"]


# In[39]:


categorical_features


# In[40]:


dataset[categorical_features].nunique()


# ### Univariate Analysis

# In[41]:


# Using the countplots to observe the counts of each categorical variable


# In[42]:


for feature in categorical_features:
    sns.countplot(feature, data = dataset, palette = "hls")
    plt.title("Countplot")
    plt.show()


# ### Bivariate analysis

# In[43]:


# Cross-tabulation of independent varibales with respect to dependent variable


# In[44]:


for feature in categorical_features:
    pd.crosstab(dataset[feature], dataset["Is_Lead"]).plot(kind="bar")
    plt.title("Crosstab plot")
    plt.show()


# ### Correlation Matrix

# In[45]:


corr = dataset.corr()


# In[46]:


corr


# In[47]:


plt.figure(figsize=(10,5))
sns.heatmap(corr, cbar = True, cmap ='viridis')


# #### There is 63% correlation between "Age" and "Vintage"

# ### Handling missing values

# In[48]:


dataset.isna().sum()


# In[49]:


dataset["Credit_Product"].value_counts()


# In[50]:


dataset["Credit_Product"].unique()


# #### Since the most frequent category is "No", the missing values are filled with the same

# In[51]:


dataset["Credit_Product"].fillna("No", inplace= True)


# In[52]:


dataset.isna().sum()


# ### Encoding the categorical variables

# In[53]:


dataset[categorical_features].nunique()


# In[54]:


categorical_features


# In[55]:


# Label encoding 'Region_Code' and 'Channel_Code'


# In[56]:


from sklearn.preprocessing import LabelEncoder


# In[57]:


encoder = LabelEncoder()


# In[58]:


# Converting the tese two variables from object type to integer type by extracting numbers


# In[59]:


def split_num(my_str):
    num = [x for x in my_str if x.isdigit()]
    num = "".join(num)

    if not num:
        num = None

    return num


# In[60]:


Rgn = []
for i in dataset['Region_Code']:
    rgn = split_num(i)
    Rgn.append(rgn)
Rgn = pd.to_numeric(Rgn)


# In[61]:


Chn = []
for i in dataset['Channel_Code']:
    chn = split_num(i)
    Chn.append(chn)
Chn = pd.to_numeric(Chn)


# In[62]:


dataset['Region_Code'] = Rgn
dataset['Channel_Code'] = Chn


# In[63]:


# Encoding other categorical variables


# In[64]:


dataset = pd.get_dummies(dataset, columns=['Gender',
 'Occupation',
 'Credit_Product',
 'Is_Active'], drop_first = True)


# In[65]:


dataset.info()


# ### Importing test dataset and applying same transformations

# In[66]:


data_test = pd.read_csv(r"C:\Users\Admin\Downloads\test_set.csv")
data_test.head()


# In[67]:


data_test.shape


# ### Handling outliers

# In[68]:


IQR_data = data_test["Avg_Account_Balance"].quantile(0.75) - data_test["Avg_Account_Balance"].quantile(0.25)
lower_bridge = data_test["Avg_Account_Balance"].quantile(0.25) - (IQR_data * 3)
upper_bridge = data_test["Avg_Account_Balance"].quantile(0.75) + (IQR_data * 3)
print(lower_bridge,'\n',upper_bridge)


# In[69]:


data_test.loc[data_test["Avg_Account_Balance"]> upper_bridge, "Avg_Account_Balance"] = upper_bridge


# In[70]:


data_test.isna().sum()


# In[71]:


data_test['Credit_Product'].fillna("No", inplace = True)


# ### Encoding the categorical variables

# In[72]:


data_test[categorical_features].nunique()


# In[73]:


# Label encoding 'Region_Code' and 'Channel_Code'


# In[74]:


encoder = LabelEncoder()


# In[75]:


# Converting the tese two variables from object type to integer type by extracting numbers


# In[76]:


Rgn = []
for i in data_test['Region_Code']:
    rgn = split_num(i)
    Rgn.append(rgn)
Rgn = pd.to_numeric(Rgn)


# In[77]:


Chn = []
for i in data_test['Channel_Code']:
    chn = split_num(i)
    Chn.append(chn)
Chn = pd.to_numeric(Chn)


# In[78]:


data_test['Region_Code'] = Rgn
data_test['Channel_Code'] = Chn


# In[79]:


# Encoding other categorical variables


# In[80]:


data_test = pd.get_dummies(data_test, columns=['Gender',
 'Occupation',
 'Credit_Product',
 'Is_Active'], drop_first = True)


# In[81]:


data_test = data_test.drop(columns=["ID"])


# In[82]:


data_test.info()


# ## Scaling

# In[83]:


from sklearn.preprocessing import MinMaxScaler


# In[84]:


scaler = MinMaxScaler()


# In[85]:


dataset_input = dataset.drop(columns= ["Is_Lead"], axis = 1)


# In[86]:


scaler.fit(dataset_input)


# In[87]:


scaled_data = pd.DataFrame(scaler.transform(dataset_input), columns = dataset_input.columns)


# In[88]:


scaled_data.head()


# In[89]:


# Scaling test data


# In[90]:


scaler.fit(data_test)


# In[91]:


scaled_test = pd.DataFrame(scaler.transform(data_test), columns = data_test.columns)


# ### Assigning Train and test

# In[92]:


X_train = scaled_data
y_train = dataset["Is_Lead"]


# In[93]:


X_test = scaled_test


# In[94]:


X_train.shape, X_test.shape


# ## Model Building

# ### Logisitic Regression

# In[95]:


from sklearn.linear_model import LogisticRegression


# In[96]:


classifier = LogisticRegression(max_iter = 200)


# In[97]:


classifier.fit(X_train, y_train)


# In[98]:


classifier.coef_


# In[99]:


classifier.predict_proba(X_train)


# In[100]:


classifier.predict_proba(X_test)


# In[101]:


pred = classifier.predict_proba(X_test)[:,1]


# In[102]:


from sklearn import metrics


# In[103]:


# calculate the fpr and tpr for all thresholds of the classification
probs = classifier.predict_proba(X_train)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_train, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[104]:


# ROC curve


# In[105]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[106]:


pred = classifier.predict_proba (X_test) [:,1]


# In[107]:


pred = pd.DataFrame(pred)


# In[108]:


pred.head()


# ### Random Forest Model

# In[109]:


from sklearn.ensemble import RandomForestClassifier


# In[110]:


clf = RandomForestClassifier(n_estimators=1000, max_depth = 4, min_samples_split=3)


# In[111]:


clf.fit(X_train, y_train)


# In[112]:


y_pred_tr = clf.predict_proba(X_train)


# In[113]:


y_pred_ts = clf.predict_proba(X_test)


# In[114]:


# Calculating fpr and tpr for all thresholds


# In[115]:


probs = clf.predict_proba(X_train)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_train, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[116]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[117]:


pred = clf.predict_proba (X_test) [:,1]


# In[118]:


pred = pd.DataFrame(pred)
pred.head()


# In[ ]:





# In[119]:


# Get numerical feature importances
importances = list(clf.feature_importances_)


# In[120]:


feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]


# In[121]:


feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)


# In[122]:


# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[ ]:





# ### XG Boost Model

# In[123]:


from xgboost import XGBClassifier


# In[124]:


model = XGBClassifier(booster = 'gbtree', n_estimators = 1000, reg_alpha = 1 )


# In[125]:


model.fit(X_train, y_train)


# In[126]:


y_train_pred = model.predict_proba(X_train)


# In[127]:


y_pred = model.predict_proba(X_test)


# In[128]:


y_pred


# In[129]:


probs = model.predict_proba(X_train)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_train, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[130]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[131]:


pred = model.predict_proba (X_test) [:,1]


# In[132]:


pred = pd.DataFrame(pred)
pred.head()


# ### XG Boost hyperparameter tuning

# In[133]:


clf = XGBClassifier(objective="binary:logistic")


# In[134]:


booster=['gbtree']
base_score=[0.3, 0.5]


# In[135]:


n_estimators = [500, 800]
max_depth = [4, 5]
learning_rate=[0.1,0.2]


# In[136]:


# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'booster':booster,
    'base_score':base_score
    }


# In[137]:


from sklearn.model_selection import RandomizedSearchCV


# In[138]:


# Set up the random search with 3-fold cross validation
random_cv = RandomizedSearchCV(estimator=clf,
            param_distributions=hyperparameter_grid,
            cv=3, n_iter=10,
            scoring = 'roc_auc',n_jobs = 3,
            verbose = 5, 
            return_train_score = True,
            random_state=24)


# In[139]:


random_cv.fit(X_train,y_train)


# In[140]:


random_cv.best_estimator_


# In[141]:


classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=1,  monotone_constraints='()',
              n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


# In[142]:


classifier.fit(X_train,y_train)


# In[143]:


y_train_pred = classifier.predict_proba(X_train)


# In[144]:


y_pred = classifier.predict_proba(X_test)


# In[145]:


y_pred


# In[146]:


probs = classifier.predict_proba(X_train)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_train, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[147]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[148]:


pred = classifier.predict_proba (X_test) [:,1]


# In[149]:


pred = pd.DataFrame(pred)
pred.head()


# ### CatBoost Classfier with encoding

# In[150]:


from catboost import CatBoostClassifier


# In[151]:


model = CatBoostClassifier(n_estimators=1000, learning_rate =0.05, depth =4, eval_metric='AUC')


# In[152]:


model.fit(X_train, y_train)


# In[153]:


y_train_pred = model.predict_proba(X_train)


# In[154]:


y_pred = model.predict_proba(X_test)


# In[155]:


y_pred


# In[156]:


probs = model.predict_proba(X_train)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_train, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[157]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[158]:


pred = model.predict_proba (X_test) [:,1]


# In[159]:


pred = pd.DataFrame(pred)
pred.head()


# In[ ]:





# In[ ]:





# ### CatBoost Classifier without encoding

# In[160]:


data = pd.read_csv(r"C:\Users\Admin\Downloads\train_set.csv")


# In[161]:


dataset = data


# In[162]:


dataset.drop(["ID"], axis=1, inplace = True)


# In[163]:


dataset.head()


# ### Handling outliers

# In[164]:


IQR_data = dataset["Avg_Account_Balance"].quantile(0.75) - dataset["Avg_Account_Balance"].quantile(0.25)
lower_bridge = dataset["Avg_Account_Balance"].quantile(0.25) - (IQR_data * 3)
upper_bridge = dataset["Avg_Account_Balance"].quantile(0.75) + (IQR_data * 3)
print(lower_bridge,'\n',upper_bridge)


# In[165]:


dataset.loc[dataset["Avg_Account_Balance"]> upper_bridge, "Avg_Account_Balance"] = upper_bridge


# In[166]:


((dataset < (Q1 - 3 * IQR)) | (dataset > (Q3 + 3 * IQR))).sum()


# ### Handling missing values

# In[167]:


dataset.isna().sum()


# #### Since the most frequent category is "No", the missing values are filled with the same

# In[168]:


dataset["Credit_Product"].fillna("No", inplace= True)


# In[169]:


dataset.isna().sum()


# In[ ]:





# In[170]:


# Similar pre processing for test data


# In[171]:


data_test = pd.read_csv(r"C:\Users\Admin\Downloads\test_set.csv")


# In[172]:


data_test.drop(["ID"], axis=1, inplace = True)


# In[173]:


data_test.head()


# ### Handling outliers

# In[174]:


IQR_data = data_test["Avg_Account_Balance"].quantile(0.75) - data_test["Avg_Account_Balance"].quantile(0.25)
lower_bridge = data_test["Avg_Account_Balance"].quantile(0.25) - (IQR_data * 3)
upper_bridge = data_test["Avg_Account_Balance"].quantile(0.75) + (IQR_data * 3)
print(lower_bridge,'\n',upper_bridge)


# In[175]:


data_test.loc[data_test["Avg_Account_Balance"]> upper_bridge, "Avg_Account_Balance"] = upper_bridge


# In[176]:


((data_test < (Q1 - 3 * IQR)) | (data_test > (Q3 + 3 * IQR))).sum()


# ### Handling missing values

# In[177]:


data_test.isna().sum()


# #### Since the most frequent category is "No", the missing values are filled with the same

# In[178]:


data_test["Credit_Product"].fillna("No", inplace= True)


# In[179]:


data_test.isna().sum()


# In[180]:


dataset.head()


# In[181]:


data_test.head()


# In[182]:


X_train = dataset.drop(["Is_Lead"], axis = 1)
y_train = dataset["Is_Lead"]
X_test = data_test


# In[183]:


X_train.head()


# In[184]:


X_test.head()


# In[185]:


data_in = X_train.drop(['Gender',
 'Region_Code',
 'Occupation',
 'Channel_Code',
 'Credit_Product',
 'Is_Active'], axis = 1)


# In[186]:


data_test = X_test.drop(['Gender',
 'Region_Code',
 'Occupation',
 'Channel_Code',
 'Credit_Product',
 'Is_Active'], axis = 1)


# In[187]:


scaler.fit(data_in)


# In[188]:


scaled_data = pd.DataFrame(scaler.transform(data_in), columns = data_in.columns)


# In[189]:


scaled_data.head()


# In[190]:


# Scaling test data


# In[191]:


scaler.fit(data_test)


# In[192]:


scaled_test = pd.DataFrame(scaler.transform(data_test), columns = data_test.columns)


# In[193]:


scaled_test.head()


# In[194]:


data1 = X_train[['Gender',
 'Region_Code',
 'Occupation',
 'Channel_Code',
 'Credit_Product',
 'Is_Active']]


# In[195]:


tr_data = pd.concat([scaled_data, data1], ignore_index =False, axis = 1)


# In[196]:


tr_data


# In[197]:


# for test data


# In[198]:


data2 = X_test[['Gender',
 'Region_Code',
 'Occupation',
 'Channel_Code',
 'Credit_Product',
 'Is_Active']]


# In[199]:


ts_data = pd.concat([scaled_test, data2], ignore_index =False, axis = 1)


# In[200]:


ts_data


# In[201]:


# Training and testing


# In[202]:


X_train = tr_data.drop(['Gender'], axis = 1)
X_test = ts_data.drop(['Gender'], axis = 1)


# In[203]:


X_test.head()


# In[204]:


categorical_features


# In[205]:


cat_features = [3,4,5,6,7]


# In[206]:


# Model building


# In[207]:


model = CatBoostClassifier(n_estimators=1000, learning_rate =0.05, loss_function='CrossEntropy',l2_leaf_reg=2 , max_depth =5, eval_metric='AUC')


# In[208]:


model.fit(X_train, y_train,cat_features=cat_features)


# In[209]:


y_train_pred = model.predict_proba(X_train)


# In[210]:


y_pred = model.predict_proba(X_test)


# In[211]:


y_pred


# In[212]:


probs = model.predict_proba(X_train)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_train, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[213]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[214]:


pred = model.predict_proba (X_test) [:,1]


# In[215]:


pred = pd.DataFrame(pred)
pred.head()


# In[ ]:





# In[ ]:





# ### Saving the predicted values along with ID and Response

# In[216]:


df = pd.DataFrame(pred, columns = ['ID', 'Is_Lead'],index= None)


# In[217]:


df['Is_Lead'] = pred


# In[218]:


df.head()


# In[219]:


df.shape


# In[220]:


data_test = pd.read_csv(r"C:\Users\Admin\Downloads\test_set.csv")


# In[221]:


Id = data_test["ID"]


# In[222]:


df["ID"] = Id


# In[223]:


df.head()


# In[224]:


df.shape


# In[225]:


df.to_csv('AV_bank_lead_prediction(7).csv', index = False) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




