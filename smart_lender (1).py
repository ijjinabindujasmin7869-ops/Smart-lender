import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("loan.csv")

print(data.head())

print("\nNumber of Rows and Columns")
print(data.shape)

print("\nColumn Names")
print(data.columns)

print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

# Fill numeric columns with median
numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_columns:
    data[col] = data[col].fillna(data[col].median())

# Fill text columns with mode
object_columns = data.select_dtypes(include=["object"]).columns

for col in object_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing Values After Filling")
print(data.isnull().sum())

for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))


print("\nDataset After Encoding")
print(data.head())

print("\n------------------------------")
print("INPUT AND OUTPUT : ")
print("------------------------------")

X = data.drop(["Loan_ID", "Loan_Status"], axis=1)

y = data["Loan_Status"]

print("\nInput Data (X)")
print(X.head())

print("\nOutput Data (y)")
print(y.head())

print("\n------------------------------")
print("TRAIN TEST SPLIT : ")
print("------------------------------")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("\n------------------------------")
print("STEP 11 : CREATE MODEL")
print("------------------------------")

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

print("Model Training Completed Successfully")

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print("\n------------------------------")
print("MODEL ACCURACY")
print("------------------------------")
print("Accuracy :", accuracy)

print("\n------------------------------")
print("LOAN PREDICTION")
print("------------------------------")

sample = X.iloc[[0]]

sample = scaler.transform(sample)

result = model.predict(sample)

if result[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")


joblib.dump(model, "smart_lender_model.pkl")

print("\nModel Saved Successfully")