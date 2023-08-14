#%%
# Modules
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Functions


# Settings
measurements  ={}
path = 'Processed_data'




#%%
# Load data into one location
total_parts = 0
for file in os.listdir(path):
    if file.endswith('.csv'):
        data = pd.read_csv(f'{path}/{file}', index_col=0)
        data = data.reset_index(drop=True)
        total_parts += int(len(data) / 50)
        measurements[file] = data

Dimensions = 6
Activity = 50
data = np.empty((total_parts, Activity, Dimensions))
outcome = np.empty((total_parts))

counter = 0
for measurement in measurements:
    file_size = len(measurements[measurement])
    for i in range(file_size//50):
        part = measurements[measurement].loc[i*50:i*50+49]
        data[counter] = part.loc[:,'Ax':'Gz'].values
        outcome[counter] = part.iloc[0]['Categories']
        counter += 1

        
num_categories = int(len(np.unique(outcome, return_counts=True)[0]))
y_one_hot = to_categorical(outcome, num_classes=num_categories)

# Build the neural network
model = Sequential([
    LSTM(64, input_shape=(Activity, Dimensions), return_sequences=True),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(num_categories, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, y_one_hot, epochs=10, batch_size=32, validation_split=0.2)

predicted_labels = model.predict(data)
y_pred = np.argmax(predicted_labels, axis=1)


# Compute confusion matrix
cm = confusion_matrix(outcome, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=["Zitten", "Fietsen", "Lopen","Overig"],
            yticklabels=["Zitten", "Fietsen", "Lopen", "Overig"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


#%%
