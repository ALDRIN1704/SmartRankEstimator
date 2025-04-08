import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# -------------------------------
# Step 1: Load the Cleaned Dataset
# -------------------------------
df = pd.read_csv("Cleaned_NIRF_2022_2023_2024.csv")
df = df.sort_values(by=["Institute Name", "Year"])

# -------------------------------
# Step 2: Filter Institutes with All 3 Years
# -------------------------------
year_counts = df.groupby("Institute Name")["Year"].nunique()
valid_names = year_counts[year_counts == 3].index
df = df[df["Institute Name"].isin(valid_names)]

print(f"✅ Institutes with all 3 years of data: {len(valid_names)}")

# -------------------------------
# Step 3: Feature Selection and Scaling
# -------------------------------
features = ['TLR(100)', 'RPC(100)', 'GO(100)', 'OI(100)', 'Perception(100)']
target = 'Score'

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df[features] = feature_scaler.fit_transform(df[features])
df[target] = target_scaler.fit_transform(df[[target]])

# -------------------------------
# Step 4: Create 3-Year Sequences for LSTM
# -------------------------------
X, y, institute_names = [], [], []

for name, group in df.groupby("Institute Name"):
    group_sorted = group.sort_values(by="Year")
    X.append(group_sorted[features].values)
    y.append(group_sorted[target].values[-1])
    institute_names.append(name)

X = np.array(X)
y = np.array(y)

print(f"✅ Total 3-year sequences created: {len(X)}")

if len(X) == 0:
    raise ValueError("❌ No valid sequences found. Please check dataset content.")

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 6: LSTM Model Definition and Training
# -------------------------------
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2)

# -------------------------------
# Step 7: Evaluation
# -------------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"🎯 Test Loss: {loss:.4f} | MAE: {mae:.4f}")

# -------------------------------
# Step 8: Save Model & Scalers
# -------------------------------
output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)

model.save(os.path.join(output_dir, "institution_lstm_model.h5"))
joblib.dump(feature_scaler, os.path.join(output_dir, "feature_scaler.pkl"))
joblib.dump(target_scaler, os.path.join(output_dir, "target_scaler.pkl"))
joblib.dump(institute_names, os.path.join(output_dir, "institute_names.pkl"))

print("✅ Model and scalers saved in 'saved_models/'")
