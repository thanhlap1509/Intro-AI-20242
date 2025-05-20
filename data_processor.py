import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import calendar
import config

#Load dữ liệu
data_path = config.dataset_used
file_name = data_path.split("/")[-1].split("_")[2]
print(f"Xử lý bộ dữ liệu {file_name}")
df = pd.read_csv(data_path)

#Chỉ giữ lại dữ liệu của trạm "Aotizhongxin"
df = df[df['station'] == file_name].drop(columns=['station'])

#Xử lý thời gian
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

#Tạo feature chu kỳ thời gian
def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

df['days_in_month'] = df.apply(lambda row: get_days_in_month(row['year'], row['month']), axis=1)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / df['days_in_month'])
df['day_cos'] = np.cos(2 * np.pi * df['day'] / df['days_in_month'])

# Loại bỏ các cột không cần thiết
df.drop(columns=['days_in_month', 'No', 'year', 'month', 'day', 'hour'], inplace=True, errors='ignore')

#Encode cột 'wd' (wind direction)
if 'wd' in df.columns:
    df['wd'] = LabelEncoder().fit_transform(df['wd'])

#Xử lý thiếu dữ liệu
df = df.dropna(subset=['PM2.5']).reset_index(drop=True)

# Tạo danh sách feature có tương quan cao với PM2.5
corr_matrix = df.corr()
pm25_corr = corr_matrix["PM2.5"].abs()

non_cyclic_time_features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
selected_features = [f for f in non_cyclic_time_features if pm25_corr.get(f, 0) > 0.1]
cyclic_time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
selected_features.extend(cyclic_time_features)

#Giữ lại các feature được chọn
df_selected = df[['datetime', 'PM2.5'] + selected_features]

#Impute + scale
imputer = SimpleImputer(strategy="median")
df_selected[selected_features] = imputer.fit_transform(df_selected[selected_features])

# Kiểm tra các cột constant (để tránh lỗi chia 0 khi chuẩn hóa)
zero_var_cols = df_selected[selected_features].nunique()[df_selected[selected_features].nunique() == 1].index.tolist()
if zero_var_cols:
  for col in zero_var_cols:
      selected_features.remove(col)

# Chuẩn hóa
scaler = MinMaxScaler()
df_selected[selected_features] = scaler.fit_transform(df_selected[selected_features])

# Đảm bảo cột 'datetime' đầu tiên
df_selected = df_selected[['datetime', 'PM2.5'] + selected_features]
print(f"Kích thước dữ liệu sau xử lý: {df_selected.shape}")

# Giá trị min, max, mean của từng cột sau khi chuẩn hoá
df_selected.min(), df_selected.max(), df_selected.mean()


# Thời gian cần chênh lệch giữa hai sample
print(df_selected.iloc[24].datetime  - df_selected.iloc[23].datetime)

# Tạo dữ liệu cho LSTM, nhưng chỉ lấy những time step liền nhau trong 24h
def create_sequences(data, timesteps, features):
    X, y = [], []
    expected_gap = pd.Timedelta('0 days 01:00:00') # nếu hai sample là liền nhau thì khoảng cách về feature datetime phải là 1 giờ

    for i in range(len(data) - timesteps + 1):
      isContinuous = 1
      for j in range(timesteps - 1): # check trong 24 sample liên tiếp kể từ i xem có phải liên tục về mặt thời gian không
        if data.iloc[i + j + 1].datetime - data.iloc[i + j].datetime != expected_gap:
          isContinuous = 0
          break
      if isContinuous:
        X.append(data.iloc[i:i+timesteps][features].values) # iloc : lấy đến giá trị trước i + timesteps
        y.append(data.iloc[i+timesteps - 1]['PM2.5']) # iloc thường lấy đúng giá trị i + timesteps nên phải lấy i + timesteps - 1
    return np.array(X), np.array(y)

timesteps = 24
X, y = create_sequences(df_selected, timesteps, selected_features)
print(f"\nX shape: {X.shape}, y shape: {y.shape}")

"""Save the processed"""

#Lưu dữ liệu vào Google Drive
np.save(f"./data_processed/X_{file_name}.npy", X)
np.save(f"./data_processed/y_{file_name}.npy", y)
print(f"Dữ liệu {file_name} đã được lưu thành công!")