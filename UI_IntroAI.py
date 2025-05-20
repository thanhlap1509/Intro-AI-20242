import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model.CNN_LSTM import CNN_LSTM
from model.GRU import GRU
from model.LSTM import LSTM

# Load the model
def load_model(model_type, checkpoint_path):
    if model_type == 'LSTM':
        input_size = 14
        hidden_size = 128
        output_size = 1
        dropout_rate = 0.55
        model = LSTM(input_size, hidden_size, output_size, dropout_rate)
    elif model_type == 'GRU':
        input_size = 14
        hidden_size = 128
        output_size = 1
        dropout_rate = 0.55
        model = GRU(input_size, hidden_size, output_size, dropout_rate)
    else:  # CNN-LSTM
        input_size = 14
        hidden_size = 96
        output_size = 1
        kernel_size = 3
        num_filters = 16
        pool_size = 2
        dropout_rate = 0.5
        model = CNN_LSTM(input_size, hidden_size, output_size, kernel_size, num_filters, pool_size, dropout_rate)

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["model_state_dict"])

    model.eval()
    return model

# Preprocess the data (now directly loads preprocessed numpy files)
def preprocess_data(x_file, y_file):
    """ Load preprocessed numpy arrays for X and y """
    X = np.load(x_file)
    y = np.load(y_file)

    return X, y

# Evaluate the model
def evaluate_model(model, data):
    X, y = data
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(X)):
            inputs = torch.tensor(X[i]).float().unsqueeze(-1)  # Add batch dimension
            labels = torch.tensor(y[i]).float()

            outputs, _ = model(inputs)
            y_pred.append(outputs[-1][0][0].item())
            y_true.append(labels.item())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.write(f"MAE: {mae:.4f}")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"R²: {r2:.4f}")

    return y_true, y_pred

# Plot the comparison
def plot_comparison(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Air Quality')
    plt.title('True vs Predicted Air Quality')
    st.pyplot(plt)

# Streamlit UI
st.title("Air Quality Prediction Demo")
st.write("Upload numpy files for X and y and select a model for air quality prediction.")

model_type = st.selectbox("Select Model", ['LSTM', 'GRU', 'CNN-LSTM'])
x_file = st.file_uploader("Upload X data (numpy file)", type=["npy"])
y_file = st.file_uploader("Upload y data (numpy file)", type=["npy"])

if x_file is not None and y_file is not None:
    # Save uploaded files temporarily to local disk
    x_path = "./data_processed/X_Aotizhongxin.npy"
    y_path = "./data_processed/y_Aotizhongxin.npy"

    with open(x_path, "wb") as f:
        f.write(x_file.getbuffer())
    
    with open(y_path, "wb") as f:
        f.write(y_file.getbuffer())

    # Load preprocessed data
    data = preprocess_data(x_path, y_path)

    # Select the best checkpoint for the chosen model
    if model_type == 'LSTM':
        checkpoint_path = r"./checkpoint/model/best_LSTM_h128_b8_l0.001_oAdam_s_cosine_wr0.1_e100.pth"
    elif model_type == 'GRU':
        checkpoint_path = r"./checkpoint/model/best_GRU_h128_b8_l0.0005_oAdam_s_cosine_wr0.2_e150.pth"
    else:
        checkpoint_path = r"./checkpoint/model/best_CNN_LSTM_h96_b8_l0.001_oAdam_s_cosine_wr0.1_e100.pth"
    
    # Load the model and evaluate
    model = load_model(model_type, checkpoint_path)
    y_true, y_pred = evaluate_model(model, data)
    
    # Plot the comparison
    plot_comparison(y_true, y_pred)
