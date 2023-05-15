from Machine import Classifier
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam
clas=Classifier()
clas.data_and_info()
clas.data_and_preprocessing()
clas.up_sampling()
x_data,y_data=clas.get_data()
x_train,x_test_and_val,y_train,y_test_and_val=train_test_split(x_data,y_data,random_state=101,train_size=0.78)
x_test,x_val,y_test,y_val=train_test_split(x_test_and_val,y_test_and_val,random_state=101,train_size=0.5)
input_shape=x_train[0].shape 
print(input_shape)
model=Sequential()
model.add(Dense(16,activation="relu",input_shape=input_shape))
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(124,activation="relu"))
model.add(Dense(124,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))
print(model.summary())
model.compile(loss="binary_crossentropy",optimizer=Adam(),metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,validation_data=(x_val,y_val))
predictions=model.predict(x_test)
print(predictions)