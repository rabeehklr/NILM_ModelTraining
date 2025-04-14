import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

torch.manual_seed(42)
np.random.seed(42)

class NILMDataset(Dataset):
    def __init__(self, X, y_status, y_power):
        self.X = torch.FloatTensor(X)
        self.y_status = torch.FloatTensor(y_status)
        self.y_power = torch.FloatTensor(y_power)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_status[idx], self.y_power[idx]

class CNNLSTMModel(nn.Module):
    def __init__(self, input_features=3, time_steps=10, num_appliances=6):
        super().__init__()
        self.time_steps = time_steps
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        cnn_output_size = 128 * time_steps  # No pooling, full sequence preserved
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        # Output heads
        self.status_head = nn.Sequential(
            nn.Linear(512, 128),  # 512 from bidirectional (256*2)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_appliances),
            nn.Sigmoid()
        )
        self.power_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_appliances),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x shape: [batch_size, input_features, time_steps]
        cnn_out = self.cnn(x)  # [batch_size, 128, time_steps]
        cnn_out = cnn_out.permute(0, 2, 1)  # [batch_size, time_steps, 128]
        lstm_out, _ = self.lstm(cnn_out)  # [batch_size, time_steps, 512]
        lstm_out = lstm_out[:, -1, :]  # Last timestep: [batch_size, 512]
        status_pred = self.status_head(lstm_out)
        power_pred = self.power_head(lstm_out)
        return status_pred, power_pred

class NILMTrainer:
    def __init__(self, dataset_path, device=None):
        self.dataset_path = dataset_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.appliances = ['fan', 'mobile charger', 'laptop charger', 'bulb', 'refrigerator', 'television']
    
    def load_and_preprocess_data(self, time_steps=10, feature_columns=['voltage', 'current', 'active_power']):
        self.data = pd.read_csv(self.dataset_path)
        features = self.data[feature_columns].values
        features_normalized = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y_status, y_power = [], [], []
        for i in range(len(features_normalized) - time_steps + 1):
            X.append(features_normalized[i:i+time_steps])
            status = [self.data[f'{app}_status'].iloc[i+time_steps-1] for app in self.appliances]
            power = [self.data[f'{app}_power'].iloc[i+time_steps-1] for app in self.appliances]
            y_status.append(status)
            y_power.append(power)
        
        X = np.array(X).transpose(0, 2, 1)  # [samples, features, time_steps]
        y_status = np.array(y_status)
        y_power = np.array(y_power)
        
        # Split
        X_train_val, X_test, y_status_train_val, y_status_test, y_power_train_val, y_power_test = train_test_split(
            X, y_status, y_power, test_size=0.2, shuffle=False
        )
        X_train, X_val, y_status_train, y_status_val, y_power_train, y_power_val = train_test_split(
            X_train_val, y_status_train_val, y_power_train_val, test_size=0.25, shuffle=False
        )
        
        self.train_dataset = NILMDataset(X_train, y_status_train, y_power_train)
        self.val_dataset = NILMDataset(X_val, y_status_val, y_power_val)
        self.test_dataset = NILMDataset(X_test, y_status_test, y_power_test)
        
        print(f"Training Set: {X_train.shape}")
        print(f"Validation Set: {X_val.shape}")
        print(f"Test Set: {X_test.shape}")
        return self
    
    def create_model(self, input_features=3, time_steps=10):
        self.model = CNNLSTMModel(input_features, time_steps, len(self.appliances)).to(self.device)
        return self
    
    def train_model(self, epochs=100, batch_size=64, learning_rate=0.001):
        wandb.init(project="NILM-Load-Disaggregation", config={"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate})
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        status_loss_fn = nn.BCELoss()
        power_loss_fn = nn.MSELoss()
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.model.train()
            epoch_train_loss = 0
            for X_batch, status_batch, power_batch in train_loader:
                X_batch = X_batch.to(self.device)
                status_batch = status_batch.to(self.device)
                power_batch = power_batch.to(self.device)
                
                optimizer.zero_grad()
                status_pred, power_pred = self.model(X_batch)
                
                status_loss = status_loss_fn(status_pred, status_batch)
                power_loss = power_loss_fn(power_pred, power_batch)
                total_loss = status_loss + power_loss  # Equal weighting
                
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += total_loss.item()
            
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, status_batch, power_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    status_batch = status_batch.to(self.device)
                    power_batch = power_batch.to(self.device)
                    status_pred, power_pred = self.model(X_batch)
                    status_loss = status_loss_fn(status_pred, status_batch)
                    power_loss = power_loss_fn(power_pred, power_batch)
                    total_loss = status_loss + power_loss
                    epoch_val_loss += total_loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            wandb.log({'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            print(f'Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('best_model.pth')
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_progress.png')
        plt.close()
        wandb.finish()
        return self
    
    def save_model(self, save_path='nilm_model.pth'):
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_state': self.scaler,
            'appliances': self.appliances
        }
        torch.save(model_state, save_path)
        print(f"Model saved to {save_path}")
        return self

def main():
    dataset_path = r'C:\Users\ASUS\Desktop\S8 PROJECT\1\nilm_synthetic_dataset_2.csv'
    trainer = (NILMTrainer(dataset_path)
               .load_and_preprocess_data(time_steps=10)
               .create_model()
               .train_model(epochs=100, batch_size=64, learning_rate=0.001)
               .save_model('nilm_modelv1.pth'))

if __name__ == "__main__":
    main()