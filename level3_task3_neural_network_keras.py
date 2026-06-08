#step 1: import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

#step 2: load data
df = pd.read_csv ("Data Set For Task/1) iris.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())

#step 3: seperate features and target
X = df.drop("species", axis=1)
y = df["species"]

#step 4: encode target column
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(le.classes_)
print(y_encoded[:10])

#step 5: scale features
scaler= StandardScaler()
x_scaled = scaler.fit_transform(X)

#step 6: split the dataset
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#step 7 : build the neural network
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])

#step 8 : compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#step 9 : train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

#step 10 : evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

#step 11 : plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("training_validation_loss.png", dpi=300)
plt.show()

#step 12 : plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig("training_validation_accuracy.png", dpi=300)
plt.show()

#step 13 : make predictions
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)
print("Predicted:", le.inverse_transform(predicted_classes[:5]))
print("Actual:", le.inverse_transform(y_test[:5]))


