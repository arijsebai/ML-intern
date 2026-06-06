# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# STEP 2: LOAD THE DATASET
df= pd.read_csv("Data Set For Task/1) iris.csv")
print("Data set shape", df.shape)
print("Data set columns names and data types:", df.dtypes)
print("Missing values:", df.isnull().sum())

# STEP 3: HANDLE MISSING DATA
print ("total missing values ", df.isnull().sum().sum())
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    if df[col].isnull().sum()> 0:
        mean_value= df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in '{col}' with mean: {mean_value:.2f}")

print ("after handling missing data ", df.isnull().sum().sum())

# STEP 4: IDENTIFY AND ENCODE CATEGORICAL VARIABLES
categorical_columns= df.select_dtypes(include=['object']).columns
if len(categorical_columns)>0:
    df_encoded = df.copy()
    label_encoders ={}
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col]=le
    df= df_encoded
    print("after encoding\n", df.head())

# STEP 5: SEPARATE FEATURES (X) AND TARGET (Y)
X = df.iloc[:, :-1]
Y = df.iloc[:, :-1]
print("X shape", X.shape)
print("X columns", X.columns)
print("Y shape", Y.shape)
print("Y columns", Y.columns)

# STEP 6: NORMALIZE / STANDARDIZE NUMERICAL FEATURES
print("Before normalizing")
print(f"Feature 1 - Min {X.iloc[:,0].min():.2f}, Max {X.iloc[:,0].max():.2f}")
print(f"Feature 1 - Min {X.iloc[:,1].min():.2f}, Max {X.iloc[:,0].max():.2f}")
scaler = StandardScaler()
x_scaled= scaler.fit_tranform(X)
X_scaled= pd.DataFrame(x_scaled, columns=X.columns)
print(X_scaled.head())

# STEP 7: SPLIT INTO TRAINING AND TESTING SETS
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# STEP 8: SAVE PREPROCESSED DATA
X_train.to_csv("X_train_preprocessed.csv", index=False)
Y_train.to_csv("Y_train_preprocessed.csv", index=False)
X_test.to_csv("X_test_preprocessed.csv", index=False)
Y_test.to_csv("Y_test_preprocessed.csv", index=False)
print("Preprocessed data saved successfully.")