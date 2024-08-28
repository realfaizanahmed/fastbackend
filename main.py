from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

app = FastAPI()

# Load and preprocess data
df = pd.read_csv('medical_cost.csv')
X = df.drop(columns=['charges'])
y = df['charges']

# Define preprocessing
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
X_preprocessed = preprocessor.fit_transform(X)

# Train models
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

lr_model = Ridge()
lr_model.fit(X_train, y_train)

fnn_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
fnn_model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
fnn_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

train_preds_combined = np.column_stack((lr_model.predict(X_train), fnn_model.predict(X_train).flatten()))
test_preds_combined = np.column_stack((lr_model.predict(X_test), fnn_model.predict(X_test).flatten()))

meta_model = LinearRegression()
meta_model.fit(train_preds_combined, y_train)

# Calculate metrics
y_test_preds = meta_model.predict(test_preds_combined)
mse = mean_squared_error(y_test, y_test_preds)
mae = mean_absolute_error(y_test, y_test_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_preds)

class UserInput(BaseModel):
    age: float
    bmi: float
    children: int
    sex: str
    smoker: str
    region: str

@app.post("/predict/")
async def predict(data: UserInput):
    try:
        user_df = pd.DataFrame([data.dict()])
        user_preprocessed = preprocessor.transform(user_df)
        lr_pred = lr_model.predict(user_preprocessed)
        fnn_pred = fnn_model.predict(user_preprocessed).flatten()
        combined_pred = np.column_stack((lr_pred, fnn_pred))
        final_pred = meta_model.predict(combined_pred)
        return JSONResponse({
            "predicted_cost": float(final_pred[0]),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)