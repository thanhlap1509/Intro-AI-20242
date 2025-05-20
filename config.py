# dataset
dataset_used = "./data_raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv"

# model param
model_type = "LSTM" # switch between GRU, LSTM, CNN_LSTM
hidden_size = 128
kernel_size = 3
num_filters = 16
pool_size = 2
output_size = 1
dropout_rate = 0.5

# training
continue_training = 0 # set to 0 to begin training, 1 for finish training
train_ratio = 0.8
batch_size = 16
optimizer_type = "AdamW"
learning_rate = 5e-4 # có thể chỉnh # Starting lr is 1e-3, modify if necessary
num_epoch = 100 # có thể chỉnh
warmup_ratio = 0.1 # có thể chỉnh
scheduler_type = "cosine"
