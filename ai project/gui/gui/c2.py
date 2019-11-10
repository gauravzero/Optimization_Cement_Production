import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt

import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

#part 2

def get_data():
    #get train data
    train_data_path =r'C:/Users/Lenovo/Desktop/train.csv'
    train = pd.read_csv(train_data_path)
    
    #get test data
    test_data_path =r'C:/Users/Lenovo/Desktop/test.csv'
    test = pd.read_csv(test_data_path)
    
    return train , test

def get_combined_data():
  #reading train data
  train , test = get_data()

  target = train.SalePrice
  train.drop(['SalePrice'],axis = 1 , inplace = True)

  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop(['index', 'Id'], inplace=True, axis=1)
  return combined, target

#PART 3 no data columns
def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

def split_combined():
    global combined
    train = combined[:296]
    test = combined[296:]

    return train , test 

def transposer(x):
    x = x.reshape(30,1).T
    return -(NN_model.predict(x))


#Load train and test data into pandas DataFrames
train_data, test_data = get_data()

#Combine train and test data to process them together
combined, target = get_combined_data()


#PART 4 collecting the rows and columns with no empty values
num_cols = get_cols_with_no_nans(combined , 'num')
cat_cols = get_cols_with_no_nans(combined , 'no_num')

#PART 5

print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))

#CORRELATION
train_data = train_data[num_cols + cat_cols]
train_data['Target'] = target

C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()

#splitting matrices
train, test = split_combined()

#neural
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

#checkpoint
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

#train
gaurav=NN_model.fit(train, target, epochs=1, batch_size=10, validation_split = 0.2, callbacks=callbacks_list)


# Load wights file of the best model :
#wights_file = 'Weights-499--5.89169.hdf5' # choose the best checkpoint 
#NN_model.load_weights(wights_file) # load it
#NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


from pyswarm import pso
plt.plot(gaurav.history['loss'])

X_predict = np.array([268.7,2.3,13.2,-0.4,9,41.8,202.8,486.8,187.7,59.7,7.3,0.5,60.9,223.9,96.2,86.1,740.3,779.5,1294.6,96.9,0.8,79.4,60.5,164.4,86.4,584.1,36.2,244.1,96.2,2.3
])
X_predict = X_predict.reshape(30,1).T
np.shape(X_predict)
print(NN_model.predict(X_predict))


###PSO PART

#def banana(x):
 #   x1 = x[0]
    #  x2 = x[1]
#    x3 = x[2]
 #   x4 = x[3]
 #   x5 = x[4]
 #   x6 = x[5]
 #   x7 = x[6]
#  #
        
#x = x.reshape(30,1).T


#return x


#def con(x):
 #   x1 = x[0]
  #  x2 = x[1]
   # return [-(x1 + 0.25)**2 + 0.75*x2]

lb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
ub = [300,5,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
xopt, fopt = pso(transposer, lb, ub)

# Optimum should be around x=[0.5, 0.76] with banana(x)=4.5 and con(x)=0

# import kivy module 
import kivy 

# this restrict the kivy version i.e 
# below this kivy version you cannot 
# use the app or software 
kivy.require("1.9.1") 

# base Class of your App inherits from the App class. 
# app:always refers to the instance of your application 
from kivy.app import App 

# creates the button in kivy 
# if not imported shows the error 
from kivy.uix.button import Button 

# class in which we are creating the button 
class ButtonApp(App): 
  
  def build(self): 
    
    btn = Button(text ="Push Me !") 
    return btn 

# creating the object root for ButtonApp() class 
root = ButtonApp() 

# run function runs the whole program 
# i.e run() method which calls the 
# target function passed to the constructor. 
root.run() 
