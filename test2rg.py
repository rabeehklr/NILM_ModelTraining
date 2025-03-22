import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
import logging
from sklearn.preprocessing import label_binarize

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('nilm_evaluation.log'),
        logging.StreamHandler()
    ]
)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_features=3, time_steps=10, num_appliances=6):
        super().__init__()
        self.time_steps = time_steps
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.status_head = nn.Sequential(
            nn.Linear(512, 128),
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
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]
        status_pred = self.status_head(lstm_out)
        power_pred = self.power_head(lstm_out)
        return status_pred, power_pred

class NILMEvaluator:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self.model = None
        self.scaler = None
        self.appliances = ['bulb', 'laptop charger', 'mobile charger']  # Only these three
        self.time_steps = 10
        self.status_threshold = 0.5
        self.min_power_threshold = 5
        self.mobile_charger_threshold = 25.0
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            logging.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            full_appliances = checkpoint.get('appliances')
            self.scaler = checkpoint.get('scaler_state')
            if self.scaler is None:
                raise ValueError("Scaler not found in checkpoint.")
            self.model = CNNLSTMModel(
                input_features=3, 
                time_steps=self.time_steps, 
                num_appliances=len(full_appliances)
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logging.info("Model and scaler loaded successfully")
            logging.info(f"Evaluating only: {self.appliances}")
            self.mobile_charger_idx = full_appliances.index('mobile charger')
            self.bulb_idx = full_appliances.index('bulb')
            self.laptop_charger_idx = full_appliances.index('laptop charger')
            self.full_appliances = full_appliances
        except Exception as e:
            logging.error(f"Error loading model or scaler: {str(e)}")
            raise

    def prepare_batch(self, features):
        if len(features) < self.time_steps:
            padding = np.zeros((self.time_steps - len(features), features.shape[1]))
            features = np.vstack((padding, features))
        elif len(features) > self.time_steps:
            features = features[-self.time_steps:]
        features_normalized = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        features_tensor = features_tensor.T.unsqueeze(0)
        return features_tensor

    def adjust_predictions(self, status_preds, power_preds, total_power):
        status_preds_np = status_preds.cpu().numpy()
        power_preds_np = power_preds.cpu().numpy()
        status = np.zeros(len(self.full_appliances), dtype=int).tolist()
        adjusted_power = np.zeros(len(self.full_appliances))

        if total_power < self.mobile_charger_threshold:
            status[self.mobile_charger_idx] = 1
            adjusted_power[self.mobile_charger_idx] = min(total_power, power_preds_np[self.mobile_charger_idx])
        else:
            status = (status_preds_np > self.status_threshold).astype(int).tolist()
            status[self.mobile_charger_idx] = 0
            adjusted_power = power_preds_np * np.array(status)
            predicted_total = np.sum(adjusted_power)
            if predicted_total == 0 and total_power > self.min_power_threshold:
                top_indices = np.argsort(status_preds_np)[-2:]
                for idx in top_indices:
                    if idx != self.mobile_charger_idx and status_preds_np[idx] > 0.3:
                        status[idx] = 1
                        adjusted_power[idx] = power_preds_np[idx]
                predicted_total = np.sum(adjusted_power)
            if predicted_total > 0 and total_power > self.min_power_threshold:
                ratio = total_power / predicted_total
                adjusted_power *= ratio
        return status, adjusted_power

    def evaluate_from_csv(self, csv_path):
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        input_cols = ['timestamp', 'voltage', 'current', 'active_power']
        gt_cols = [f"{app}_status" for app in self.appliances] + [f"{app}_power" for app in self.appliances]
        required_cols = input_cols + gt_cols
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns. Expected: {required_cols}")

        y_true_status = {app: [] for app in self.appliances}
        y_pred_status = {app: [] for app in self.appliances}
        status_probs = {app: [] for app in self.appliances}
        
        feature_buffer = np.zeros((self.time_steps, 3))
        
        for idx, row in df.iterrows():
            features = np.array([row['voltage'], row['current'], row['active_power']])
            total_power = row['active_power']
            feature_buffer[:-1] = feature_buffer[1:]
            feature_buffer[-1] = features
            X = self.prepare_batch(feature_buffer)
            with torch.no_grad():
                status_preds, power_preds = self.model(X)
            status, _ = self.adjust_predictions(status_preds[0], power_preds[0], total_power)
            for app in self.appliances:
                app_idx = self.full_appliances.index(app)
                y_true_status[app].append(int(row[f"{app}_status"]))
                y_pred_status[app].append(status[app_idx])
                status_probs[app].append(float(status_preds[0][app_idx]))
        
        output_dir = r"c:\Users\ASUS\Desktop\Current\evaluation_plots"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory set to: {output_dir}")
        
        self.plot_classification_metrics(y_true_status, y_pred_status, status_probs, output_dir)

    def plot_classification_metrics(self, y_true_status, y_pred_status, status_probs, output_dir):
        try:
            performance_metrics = {}
            # Color scheme for appliances (consistent across all plots)
            appliance_colors = {
                'mobile charger': '#FFA500',  # Orange
                'laptop charger': '#008000',  # Green
                'bulb': '#FF0000'            # Red
            }
            
            # Generate individual confusion matrices and compute metrics
            for app in self.appliances:
                y_true = np.array(y_true_status[app])
                y_pred = np.array(y_pred_status[app])
                probs = np.array(status_probs[app])
                
                # Compute metrics
                report = classification_report(y_true, y_pred, output_dict=True)
                y_true_bin = label_binarize(y_true, classes=[0, 1])
                fpr, tpr, _ = roc_curve(y_true_bin, probs)
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y_true, probs)
                pr_auc = auc(recall, precision)
                
                performance_metrics[app] = {
                    'accuracy': report['accuracy'],
                    'precision': report['1']['precision'] if '1' in report else 0,
                    'recall': report['1']['recall'] if '1' in report else 0,
                    'f1_score': report['1']['f1-score'] if '1' in report else 0,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc
                }
                
                # Plot confusion matrix
                plt.figure(figsize=(6, 5))
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Off', 'On'],
                        yticklabels=['Off', 'On'])
                plt.title(f'{app} Confusion Matrix')
                plt.xlabel('Predicted Status')
                plt.ylabel('True Status')
                plot_path = os.path.join(output_dir, f"{app}_confusion_matrix.png")
                plt.savefig(plot_path)
                plt.close()
                if os.path.exists(plot_path):
                    logging.info(f"Saved confusion matrix for {app} to {plot_path}")
                else:
                    logging.error(f"Failed to save confusion matrix for {app} to {plot_path}")
            
            # Plot separate performance bar charts for each appliance
            metrics_to_plot = ['precision', 'recall', 'f1_score']
            for app in self.appliances:
                plt.figure(figsize=(6, 5))
                labels = ['Precision', 'Recall', 'F1-Score']
                values = [performance_metrics[app][metric] for metric in metrics_to_plot]
                
                bars = plt.bar(labels, values, color=appliance_colors[app], width=0.5)
                
                plt.title(f'Performance Metrics for {app}')
                plt.xlabel('Metrics')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=appliance_colors[app], label=app)]
                plt.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f"{app}_performance_metrics.png")
                plt.savefig(plot_path)
                plt.close()
                if os.path.exists(plot_path):
                    logging.info(f"Saved performance metrics plot for {app} to {plot_path}")
                else:
                    logging.error(f"Failed to save performance metrics plot for {app} to {plot_path}")
            
            # Log metrics
            logging.info("Classification Performance Metrics:")
            for app, metrics in performance_metrics.items():
                logging.info(f"\n{app} Metrics:")
                for metric, value in metrics.items():
                    logging.info(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            logging.error(f"Failed to plot classification metrics: {str(e)}")
            raise

def main():
    try:
        model_path = 'nilm_modelv2.pth'
        csv_path = 'test_data2.csv'
        output_dir = r"c:\Users\ASUS\Desktop\Current\evaluation_plots"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Starting evaluation with model: {model_path}, data: {csv_path}")
        
        evaluator = NILMEvaluator(model_path)
        evaluator.evaluate_from_csv(csv_path)
        logging.info("Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    main()