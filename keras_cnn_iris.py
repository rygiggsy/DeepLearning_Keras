'''
Trains a simple CNN on the iris dataset

~98.7% test accuracy after 100 epochs
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt

import seaborn as sns

df = sns.load_dataset("iris")

np.random.seed(123)  # for reproducibility

#- Show sample data
#df.head()

# Setup train and test sets
data_all = df.values[:,:4]
label_all = df.values[:, 4]
X_train, X_test, y_train, y_test = train_test_split(data_all, label_all, train_size=0.5, random_state=0)


# Initial Data Sanity Check
#print X_train.shape
#print Y_train[:10]

sns.pairplot(df, hue="species")

#====== Preprocess Data Labels ======
def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

Y_train = one_hot_encode_object_array(y_train)
Y_test = one_hot_encode_object_array(y_test)


#print Y_train.shape
#print Y_train[:10]

#====== Model ======
# We have four features and three classes, so the input layer must have four units, and the output layer must have three units. 
# We're only going to have one hidden layer for this project, and we'll give it 16 units.
model = Sequential()

#The next two lines define the size input layer (input_shape=(4,), and the size and activation function of the hidden layer
model.add(Dense(16, input_shape=(4,), activation='sigmoid'))
# now: model.output_shape == (None, 16)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model on training Data
model.fit(X_train, Y_train, batch_size=1, nb_epoch=100, verbose=0)                  
                  
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy = {:.3f}".format(accuracy))    
# Accuracy = 0.987        
