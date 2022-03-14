from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import numpy as np
import pandas as pd
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle

############
# model.py #
############

# This file is meant to serve as the model generation file where upon being run, a .pkl file will be generated which can then be used in main to generate results.
# This eliminates the need to re-train the model each time we make a change to the webpage.

objects = {
    0: "Bowtie",
    1: "Broom",
    2: "Crown",
    3: "EiffelTower",
    4: "HotAirBalloon",
    5: "HousePlant",
    6: "Bed",
    7: "Cat",
    8: "Couch",
    9: "Dog",
    10: "Hand",
    11: "Hat",
    12: "Tractor"
}

data = pd.DataFrame()

# Load data from all npy files
for object in objects:

    # Load the numpy file
    object_data = None
    if exists(f"./data/{objects[object]}.npy"):
        object_data = np.load(f"./data/{objects[object]}.npy")
    else:
        object_data = np.load(
            f"./DoodleClassifierModel/data/{objects[object]}.npy")

    # Add labels to data
    temp = pd.DataFrame(object_data)
    temp["Label"] = object

    # Append object data to main dataframe
    data = pd.concat([data, temp], ignore_index=True)


# Train test validation split
x_train, x_test, y_train, y_test = train_test_split(
    data.loc[:, data.columns != "Label"], data["Label"], test_size=0.33, random_state=69)

model = RFC(n_estimators=100, max_depth=None, random_state=420)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

def show_confusion_matrix():
    # Create labels
    labels = list(objects.keys())

    # Initialize confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=predictions, labels=labels)
    print(cm)

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')

    fig.colorbar(cax)

    # "If you have more than a few categories, Matplotlib decides to label the axes incorrectly - you have to force it to label every cell." - https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels([''] + list(objects.values()))
    ax.set_yticklabels([''] + list(objects.values()))
    ax.tick_params(labelrotation=45)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Draw figure
    plt.show()

# Show confusion matrix
# show_confusion_matrix()

# Save the model in a .pkl
import pickle

with open('model.pkl','wb') as f:
    pickle.dump(model, f)

if exists("./model.pkl"):
    print("Successfully generated model.pkl!")