import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

car_df = pd.read_csv('quikr_car.csv')

#QUALITY

# year has many non-year values
# year object to int
# price has ask for price
# kms_driven has kms with integers
# kms_driven object to int
# kms_driven has nan values
# fuel_type has nan values
# keep first 3 words of name of car

backup_df = car_df.copy()

car_df = car_df.dropna()

car_df = car_df[car_df['year'].str.isnumeric()]
car_df['year'] = car_df['year'].astype(int)

car_df = car_df[car_df['Price']!="Ask For Price"]
car_df['Price'] = car_df['Price'].str.replace(',','').astype(int)

car_df['kms_driven'] = car_df['kms_driven'].str.replace(',','')
car_df['kms_driven'] = car_df['kms_driven'].str.replace('kms','').astype(int)

car_df['fuel_type'] = car_df['fuel_type'].fillna(True)

car_df['name'] = car_df['name'].str.split(' ').str.slice(0,3).str.join(' ')

car_df = car_df[car_df['Price']<6e6].reset_index(drop=True)

car_df.to_csv("Cleaned_Car.csv")

x = car_df.drop(columns='Price')
y = car_df['Price']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2,random_state=661)

ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type',]])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']), remainder='passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
#error = r2_score(y_test,y_pred)
#print('R-squared score:', error)

pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
