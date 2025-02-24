import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

# 確保 models 資料夾存在
if not os.path.exists('models'):
    os.makedirs('models')

# EarlyStopping 和 ModelCheckpoint 回調函數
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss')

callbacks = [early_stopping, checkpoint]

# 載入訓練與測試資料
train_data = pd.read_csv('Google_stock_price_train.csv')
test_data = pd.read_csv('Google_stock_price_test.csv')

# 轉換日期格式
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

# 日期排序並設為索引
train_data.sort_values(by='Date', inplace=True)
test_data.sort_values(by='Date', inplace=True)
train_data.set_index('Date', inplace=True)
test_data.set_index('Date', inplace=True)

# 特徵選擇與目標設定
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'

# 處理數據中的逗號，轉換數據類型 (若數據為字串形式)
for feature in features:
    train_data[feature] = train_data[feature].replace(',', '', regex=True).astype(float)
    test_data[feature] = test_data[feature].replace(',', '', regex=True).astype(float)

# 資料縮放 (MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[features])
scaled_test_data = scaler.transform(test_data[features])

# 保存縮放器，方便日後反轉數據
joblib.dump(scaler, 'models/scaler.pkl')

# 將縮放後的資料轉為 DataFrame
scaled_train_data = pd.DataFrame(scaled_train_data, columns=features, index=train_data.index)
scaled_test_data = pd.DataFrame(scaled_test_data, columns=features, index=test_data.index)

# 預測目標
train_target = train_data[target]
test_target = test_data[target]

# LSTM 需要的滑動窗口 (Sliding Window)
PAST_DAYS = 60

def create_sequences(data, target, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i - window_size:i].values)  # 過去 window_size 天作為輸入
        y.append(target.iloc[i])  # 預測目標
    return np.array(X), np.array(y)

# 創建訓練與測試集
X_train, y_train = create_sequences(scaled_train_data, train_target, PAST_DAYS)

# 確認測試資料長度足夠
if len(scaled_test_data) <= PAST_DAYS:
    print("錯誤: 測試集資料不足，請增加測試數據或檢查滑動窗口大小。")
    exit()

X_test, y_test = create_sequences(scaled_test_data, test_target, PAST_DAYS)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

assert len(X_test.shape) == 3
assert len(y_test.shape) == 1

# 建立 LSTM 模型
model = Sequential()

# 第一層 LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(PAST_DAYS, len(features))))
model.add(Dropout(0.2))

# 第二層 LSTM
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# 輸出層
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)

# 保存最終模型
model.save('models/final_model.keras')

# 評估模型
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# 進行預測
predictions = model.predict(X_test)

# 反轉縮放 (僅針對預測的目標欄位)
def inverse_transform(scaler, data, n_features):
    # 創建空矩陣補齊形狀
    temp = np.zeros((data.shape[0], n_features))
    temp[:, -1] = data[:, 0]
    return scaler.inverse_transform(temp)[:, -1]

predictions_actual = inverse_transform(scaler, predictions, len(features))
y_test_actual = inverse_transform(scaler, y_test.reshape(-1, 1), len(features))

# 繪圖比較預測與實際股價
plt.figure(figsize=(14, 6))
plt.plot(test_data.index[-len(y_test_actual):], y_test_actual, color='red', label='Real Google Stock Price')
plt.plot(test_data.index[-len(predictions_actual):], predictions_actual, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
