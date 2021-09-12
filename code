# deeplearning1
Create basic model
## There is linear basic formula to convert celcius to fahrenheit
## multiply by 1.8 (or 9/5) and add 32
## F= C*1.8+32
## I WANT TO ACHIEVE CREATE MODEL WITHOUT USING THAT FORMULA

#all these are basic libraries to create model
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense 

#I have used celcius2fahrenheit converter online to collect data:
celcius_degree= np.array([-40,-15,-10,0,8,15,22,38,40,59,80])
fahrenheit_degree= np.array([-40,5,14,32,46,59,72,100,104,138.20,176])

#To create model I have function as TrainValidateModel().
# Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
#1 layer
#after compiled model, it shoul be fit by using celcius_degree array and fahrenheit_degree array hundered times
#this function returns created model

def TrainValidateModel():
  model = keras.Sequential()
  model.add(Dense(1))
  model.compile(loss="mean_squared_error",optimizer=keras.optimizers.Adam(0.1))
  model.fit(celcius_degree,fahrenheit_degree,epochs=1000, verbose=False)
  return model
   
#to predict fahrenheit I used trained model
#I have function for that
  
def PredictCelcius(model,celcius_degree):
  return model.predict(np.array([celcius_degree]))
  
#NOW I CREATE MODEL WITH FUNCTION:
model= TrainValidateModel()
  
#ASK USER FOR CELCIUS TO CONVERT FAHRENHEIT:
prediction_celcius=int(input("enter c value to convert F  "))
print(PredictCelcius(model,prediction_celcius))


##LETS COMPARE THE VALUE 
def Calculate2Fahrenheit(celcius):
  return celcius*1.8+32
Calculate2Fahrenheit(prediction_celcius)

#lets check weights bias
model.weights

#model gives that bias is 31.58 and w1 is 1.80 , that is almost correct and model is created succesfully
