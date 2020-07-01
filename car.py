import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pickle
df=pd.read_csv("car data.csv")

#print(df.head())
#print(df.shape)
#print(df['Seller_Type'].unique())
#print(df['Transmission'].unique())
#print(df['Owner'].unique())
#print(df.isnull().sum())
#print(df.describe())
#print(df.columns)
fdf=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
#print(fdf.shape)
fdf['Current_Year']=2020
#print(fdf.shape)
fdf['no_year']=fdf['Current_Year']-fdf['Year']
#print(fdf['no_year'])
fdf.drop(['Year','Current_Year'],axis=1,inplace=True)
#print(fdf.columns)
fdf=pd.get_dummies(fdf,drop_first=True)

#print(fdf.columns)
#print(fdf.corr())
X=fdf.iloc[:,1:]
y=fdf.iloc[:,0]
#print(X.head)
#print(y.head)
model=ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
#fea=pd.Series(model.feature_importances_,index=X.columns)
#fea.nlargest(5).plot('barh')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
n_estimators=[int(x)for x in np.linspace(start=100,stop=1200,num=12)]
max_features=['auto','sqrt']
max_depth=[int (x) for x in np.linspace(5,30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]

random_grid={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth':max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf}
print(random_grid)
rf=RandomForestRegressor()
random=RandomizedSearchCV(rf, param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42)
random.fit(X_train,y_train)
random.score(X_test,y_test)
file= open('random_forest_regression_model.pkl','wb')
pickle.dump(random,file)
