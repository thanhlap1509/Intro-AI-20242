import numpy as np # linear algebra
import matplotlib.pyplot as plt
import importlib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from transformers import get_scheduler
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchmetrics.regression import R2Score
from tqdm.auto import tqdm
import config 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

data_path = config.dataset_used
file_name = data_path.split("/")[-1].split("_")[2]

X = np.load(f"./data_processed/X_{file_name}.npy")
y = np.load(f"./data_processed/y_{file_name}.npy")

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
X_tensor = torch.unsqueeze(X_tensor, dim = -1)

y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

train_ratio = config.train_ratio
train_len = int(X_tensor.shape[0] * train_ratio)
test_len = X_tensor.shape[0] - train_len

print(f"Train size is: {train_len}")
print(f"Test size is: {test_len}")

dataset = TensorDataset(X_tensor, y_tensor)
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

print(f"Train dataset size is: {len(train_dataset)}")
print(f"Test data size is: {len(test_dataset)}")

batch_size = config.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

input_size = X_tensor.shape[2]
hidden_size = config.hidden_size 
kernel_size = config.kernel_size
num_filters = config.num_filters
pool_size = config.pool_size
output_size = config.output_size
dropout_rate = config.dropout_rate

print(f"Input, hidden, output size: {input_size, hidden_size, output_size}")

loss_fn = torch.nn.MSELoss()

model_type = config.model_type 
module = importlib.import_module(f"model.{model_type}")
ModelClass = getattr(module, model_type)
if model_type == "CNN_LSTM":
    model = ModelClass(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    kernel_size = kernel_size,
                    num_filters = num_filters,
                    pool_size = pool_size,
                    dropout_rate = dropout_rate)
else:  
    model = ModelClass(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    dropout_rate = dropout_rate)  
model.to(device)
print(f"Training model of type {type(model)}")

# Import optimizer
optimizer_type = config.optimizer_type
optimizer_class = getattr(importlib.import_module("torch.optim"), optimizer_type)
learning_rate = config.learning_rate
optimizer = optimizer_class(model.parameters(), lr=learning_rate)
print(f"Using optimizer type {optimizer_type}")

# Training arguments
num_epoch = config.num_epoch
warmup_ratio = config.warmup_ratio
total_step = num_epoch * len(train_loader)
warmup_step = int(warmup_ratio * total_step)
scheduler_type = config.scheduler_type
scheduler = get_scheduler(scheduler_type, optimizer = optimizer, num_warmup_steps = warmup_step, num_training_steps = total_step)
print(f"Using scheduler type {scheduler_type} with warmup ratio {warmup_ratio}")
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

loss_path = f"./checkpoint/loss/loss_{model_type}_h{hidden_size}_b{batch_size}_l{learning_rate}_o{optimizer_type}_s_{scheduler_type}_wr{warmup_ratio}_e{num_epoch}.txt"
best_model_path = f"./checkpoint/model/best_{model_type}_h{hidden_size}_b{batch_size}_l{learning_rate}_o{optimizer_type}_s_{scheduler_type}_wr{warmup_ratio}_e{num_epoch}.pth"
current_model_path = lambda epoch: f"./checkpoint/model/epoch_{epoch + 1}_{model_type}_h{hidden_size}_b{batch_size}_l{learning_rate}_o{optimizer_type}_s_{scheduler_type}_wr{warmup_ratio}_e{num_epoch}.pth"

with open(loss_path, 'w') as f:
    f.writelines("Epoch, training loss, development loss\n")
    f.close()
    
continue_training = config.continue_training

if continue_training:
    checkpoint_path = f"./checkpoint/model/best_{model_type}_h{hidden_size}_b{batch_size}_l{learning_rate}_o{optimizer_type}_s_{scheduler_type}_wr{warmup_ratio}_e{num_epoch}.pth"
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']

    content = ""
    print(f"Continue training from checkpoint {checkpoint_path} ")
    with open(f"./checkpoint/loss/loss_{model_type}_h{hidden_size}_b{batch_size}_l{learning_rate}_o{optimizer_type}_s_{scheduler_type}_wr{warmup_ratio}_e{num_epoch}.txt", "r") as f:
        content = f.readlines()
        f.close()
        
    with open(loss_path, 'w') as f:
        f.writelines(content)
        f.close()
else: 
    start_epoch = 0

min_dev_loss = 1e10
for epoch in range(start_epoch, num_epoch):
    model.train()
    train_losses = []
    
    for X_batch, y_batch in tqdm(train_loader):
        X_batch.to(device)
        y_batch.to(device)

        # Compute batch loss
        batch_preds = []
        batch_refs = []
        actual_batch_size = X_batch.shape[0]
        for i in range(actual_batch_size):
            inputs = X_batch[i]
            label = y_batch[i]
            outputs, _ = model(inputs)
            batch_preds.append(outputs[-1][0][0]) # extract output of final timestep of the 24 time steps
            batch_refs.append(label) # extract label of the final timestep in X
            
        # Calculate loss
        batch_preds = torch.stack(batch_preds).to(device)
        batch_refs = torch.stack(batch_refs).to(device)
        
        loss = loss_fn(batch_preds, batch_refs)
        train_losses.append(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # Validation
    model.eval()
    dev_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch.to(device)
            y_batch.to(device)
                
            # Compute batch loss
            batch_preds = []
            batch_refs = []
            actual_batch_size = X_batch.shape[0]
            for i in range(actual_batch_size):
                inputs = X_batch[i]
                label = y_batch[i]
                outputs, _ = model(inputs)
                batch_preds.append(outputs[-1][0][0]) # extract output of final timestep of the 24 time steps
                batch_refs.append(label) # extract label of the final timestep in X
                
            # Calculate loss
            batch_preds = torch.stack(batch_preds).to(device)
            batch_refs = torch.stack(batch_refs).to(device)
            
            loss = loss_fn(batch_preds, batch_refs)
            dev_losses.append(loss)

    train_losses = torch.stack(train_losses).to(device)
    dev_losses = torch.stack(dev_losses).to(device)
    avg_train_loss = train_losses.mean()
    avg_dev_loss = dev_losses.mean()

    print(f"Epoch {epoch + 1}/{num_epoch}, Train loss: {avg_train_loss: .4f}, Dev loss: {avg_dev_loss: .4f}")
    with open(loss_path, "a") as f:
        f.write(f"{epoch + 1}, {avg_train_loss: .4f}, {avg_dev_loss: .4f}\n")
        f.close()
        
    if avg_dev_loss < min_dev_loss:
        print("Saving model...")
        min_dev_loss = avg_dev_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, best_model_path)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, current_model_path(epoch))

model.load_state_dict(torch.load(best_model_path)["model_state_dict"])
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader):
        X_batch.to(device)
        y_batch.to(device)
        
        # Compute sample in batch output
        actual_batch_size = X_batch.shape[0]
        for i in range(actual_batch_size):
            inputs = X_batch[i]
            label = y_batch[i]
            outputs, _ = model(inputs)
            y_pred.append(outputs[-1][0][0]) # extract output of final timestep of the 24 time steps
            y_true.append(label) # extract label of the final timestep in X
            
    y_pred = torch.stack(y_pred).to(device)
    y_true = torch.stack(y_true).to(device)
    
    if (epoch + 1) == num_epoch:
        print("\n\nEvaluation Metrics:")
        print(f"MSE: {torch.nn.MSELoss()(y_pred, y_true):.4f}")
        print(f"MAE: {torch.nn.L1Loss()(y_pred, y_true):.4f}")
        print(f"R²:  {R2Score()(y_pred, y_true).item():.4f}")

plt.plot(y_true.cpu().detach().numpy(), color='r', label='true value')
plt.plot(y_pred.cpu().detach().numpy(), color='b', label='predicted value')
plt.xlabel("Steps")
plt.ylabel("Loss values")
plt.title("True value vs predicte value across eval set")
plt.legend()
plt.grid(True)
plt.show()
plt.clf()