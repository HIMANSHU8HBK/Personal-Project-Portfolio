#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures


# In[7]:


games = pd.read_csv('games.csv',low_memory=False,encoding= 'unicode_escape')
details = pd.read_csv('games_details.csv',low_memory=False,encoding= 'unicode_escape')
teams = pd.read_csv('teams.csv',low_memory=False,encoding= 'unicode_escape')
players = pd.read_csv('players.csv',low_memory=False,encoding= 'unicode_escape')
ranking = pd.read_csv('ranking.csv',low_memory=False,encoding= 'unicode_escape')


# In[8]:


def get_labels(ranking):
    temp = ranking.copy(deep=False)
    temp = temp.groupby(['TEAM_ID','SEASON_ID'])['G','W'].max()
    temp = pd.DataFrame(temp)
    temp.reset_index(inplace=True)
    drops = []
    for i in range(len(temp)):
        if temp.iloc[i,1] / 10000 > 2:
            temp.iloc[i,1] = temp.iloc[i,1] % 10000
        else:
            drops.append(i)
            continue;
        if (temp.iloc[i,2] != 82):
            drops.append(i)
    for i in range(len(drops)):
        temp.drop([drops[i]], inplace=True)
    temp.reset_index(inplace=True)
    temp.drop(columns=['index'], inplace=True)
    temp.drop(columns=['G'], inplace=True)
#     temp = pd.merge(temp, ranking, how='left', left_on=['TEAM_ID','STANDINGSDATE'], right_on = ['TEAM_ID','STANDINGSDATE'])
#     temp.drop(columns=['STANDINGSDATE','LEAGUE_ID','SEASON_ID_y','CONFERENCE','TEAM','G','W','L','HOME_RECORD','ROAD_RECORD','RETURNTOPLAY'], inplace=True)
    return temp


# In[9]:


labels = get_labels(ranking)
labels


# In[10]:


def get_features(games, details):
    temp = pd.merge(games, details, how='left', left_on=['GAME_ID'], right_on = ['GAME_ID'])
    temp = temp[['TEAM_ID','SEASON','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA'
             ,'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS']]
    temp = temp.groupby(['TEAM_ID','SEASON'])['FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA'
             ,'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS'].sum()
    temp = pd.DataFrame(temp)
    next_season = []
    temp.reset_index(inplace=True)
    for i in range(len(temp)):
        next_season.append(temp.iloc[i,1] + 1)
    temp['NEXT_SEASON'] = next_season
    return temp


# In[11]:


features = get_features(games, details)
features


# In[12]:


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


# In[13]:


def get_data(ranking, games, details):
    labels = get_labels(ranking)
    features = get_features(games, details)
    data = pd.merge(labels, features, how='left', left_on=['TEAM_ID','SEASON_ID'], right_on = ['TEAM_ID','NEXT_SEASON'])
    data.drop(columns=['SEASON_ID','SEASON'], inplace=True)
    data.dropna(inplace=True)
    data = swap_columns(data, 'W', 'NEXT_SEASON')
    data = data.astype({'NEXT_SEASON': 'int64'})
    data.rename(columns={'W' : 'NEXT_W'}, inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)
    return data


# In[14]:


data = get_data(ranking, games, details)
data


# In[15]:


data.to_csv('nba_data.csv')


# In[16]:


def scale_data(data):
    temp = data.copy(deep=False)
    std_slc = StandardScaler()
    preprocess = std_slc.fit_transform(temp[['FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA'
             ,'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS']])
#     preprocess = preprocessing.normalize(temp[['FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA'
#              ,'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS']])
    data_scaled = pd.DataFrame(preprocess, columns=['FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM'
                    ,'FTA','FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS'])
    data_scaled.insert(0,'TEAM_ID',temp[['TEAM_ID']])
    data_scaled.insert(1,'NEXT_SEASON',temp[['NEXT_SEASON']])
    data_scaled.insert(21,'NEXT_W',temp[['NEXT_W']])
    return data_scaled


# In[17]:


data_scaled = scale_data(data)
data_scaled


# In[18]:


data_scaled.to_csv('nba_data_scaled.csv')


# In[19]:


def split_data_X_y(data):
    temp = data.copy(deep=False)
    temp.drop(columns=['TEAM_ID','NEXT_SEASON'], inplace=True)
    X = data.iloc[:,2:].copy(deep=False)
    X.drop(columns=['NEXT_W'], inplace=True)
    y = data.iloc[:,-1:].copy(deep=False)
    return X, y


# In[20]:


def split_data_train_test(data):
    temp = data.copy(deep=False)
    temp.drop(columns=['TEAM_ID','NEXT_SEASON'], inplace=True)
    X = data.iloc[:,2:].copy(deep=False)
    X.drop(columns=['NEXT_W'], inplace=True)
    y = data.iloc[:,-1:].copy(deep=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)
    return X_train, X_test, y_train, y_test


# In[21]:


def rmse(y_test, y_pred):
    return mean_squared_error(y_test, y_pred, squared=False)


# In[22]:


X, y = split_data_X_y(data_scaled)


# In[23]:


X_train, X_test, y_train, y_test = split_data_train_test(data_scaled)


# In[24]:


def cross_validation(model, _X, _y, _cv=5):
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = make_scorer(rmse, greater_is_better=True)
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training RMSE": results['train_score'],
              "Mean RMSE": results['train_score'].mean(),
              "Validation RMSE": results['test_score'],
              "Mean Validation RMSE": results['test_score'].mean()
              }


# In[25]:


model_lr = LinearRegression()
print(cross_validation(model_lr, X, y))


# In[26]:


model_lasso = linear_model.Lasso(alpha=0.1)
print(cross_validation(model_lasso, X, y))


# In[27]:


model_svm = SVR(kernel = 'linear').fit(X_train, y_train)
print(cross_validation(model_svm, X, y))


# In[28]:


model_tree = RandomForestRegressor(random_state=2023)
print(cross_validation(model_tree, X, y))


# In[29]:


model_lasso = linear_model.Lasso(alpha=0.1)
model_lasso.fit(X_train, y_train)
y_pred = model_lasso.predict(X_test)
print(mean_squared_error(y_test, y_pred, squared=False))
print(model_lasso.coef_)


# In[30]:


def feature_selection(data):
    temp = data.copy(deep=False)
    temp.drop(columns=['FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','AST','BLK','PTS'], inplace=True)
    return temp;


# In[31]:


data_selected = feature_selection(data_scaled)
X_new, y_new = split_data_X_y(data_selected)
X_new_train, X_new_test, y_new_train, y_new_test = split_data_train_test(data_selected)
data_selected


# In[32]:


model_lr_new = LinearRegression()
print(cross_validation(model_lr_new, X_new, y_new))


# In[33]:


model_lasso_new = linear_model.Lasso(alpha=0.1)
print(cross_validation(model_lasso_new, X_new, y_new))


# In[34]:


model_svm_new = SVR(kernel = 'linear')
print(cross_validation(model_svm_new, X_new, y_new))


# In[35]:


model_tree_new = RandomForestRegressor(random_state=2023)
print(cross_validation(model_tree_new, X_new, y_new))


# In[36]:


model_lasso_new = linear_model.Lasso(alpha=0.1)
model_lasso_new.fit(X_new_train, y_new_train)


# In[37]:


def get_2017(data):
    temp = data.copy(deep=False)
    drop_list = []
    for i in range(len(temp)):
        if temp.iloc[i, 1] != 2018:
            drop_list.append(i)
            
    temp.drop(drop_list, inplace=True)
    temp.reset_index(inplace=True)
    temp.drop(columns=['index'], inplace=True)
    return temp


# In[38]:


data_2017 = get_2017(data_scaled)
data_2017


# In[39]:


data_2017_selected = feature_selection(data_2017)
X_2017, y_2018 = split_data_X_y(data_2017_selected)


# In[40]:


y_2018_pred = model_lasso_new.predict(X_2017)
print(np.round(y_2018_pred))


# In[41]:


team_id_name = {}
for i in range(len(teams)):
    team_id_name[teams.iloc[i, 1]] = teams.iloc[i, 5]
print(team_id_name)


# In[42]:


name_list = []
for i in range(len(data_2017)):
    name_list.append(team_id_name[data_2017.iloc[i, 0]])
print(name_list)


# In[43]:


prediction_dict_2018 = {"team_name": name_list, "wins_pred_2018": np.round(y_2018_pred), 'wins_2018': y_2018['NEXT_W']}
prediction_df_2018 = pd.DataFrame(prediction_dict_2018)
prediction_df_2018.sort_values(by='wins_pred_2018', ascending=False, inplace=True)
prediction_df_2018 = prediction_df_2018.astype({'wins_pred_2018': 'int64'})
prediction_df_2018.reset_index(inplace=True)
prediction_df_2018.drop(columns=['index'], inplace=True)


# In[44]:


prediction_df_2018


# In[45]:


prediction_df_2018.to_csv('nba_data_2018_prediction.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




