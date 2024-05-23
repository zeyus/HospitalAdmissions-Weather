import json
import logging
import math
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from common.config import DATA_DIR, FIGURE_DIR, read_data, prepare_data, MODEL_DIR
from common import plotting
import torch, torch.nn as nn, torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


gnuplot_available = plotting.is_gnuplot_available()
gp: plotting.gp
if gnuplot_available:
    gp = plotting.prepare_plot()

class EarlyStopping:
    def __init__(self, patience:int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: None | float = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss: float):
        score = -loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class AdmissionsLSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            hidden_layer_size: int = 10,
            dense_size: int = 128,
            output_size: int = 1,
            output_seq_len: int = 1,
            lstm_layers = 2,
            dropout: float = 0.2,
            device: torch.device = torch.device('cpu'),
            dtype=torch.float32):
        super().__init__()
        self.device = device
        self.lstm_layers = lstm_layers
        self.hidden_layer_size = hidden_layer_size
        self.output_seq_len = output_seq_len
        self.dtype = dtype
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=lstm_layers, batch_first=True, dropout=dropout, device=device, dtype=dtype)
        self.bn_lstm = nn.BatchNorm1d(hidden_layer_size, device=device, dtype=dtype)
        self.linear1 = nn.Linear(hidden_layer_size, dense_size, device=device, dtype=dtype)
        self.bn_linear = nn.BatchNorm1d(dense_size, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dense_size, output_size * output_seq_len, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_seq):
        hs = torch.zeros(self.lstm_layers, input_seq.size(0), self.hidden_layer_size, device=self.device, dtype=self.dtype)
        cs = torch.zeros(self.lstm_layers, input_seq.size(0), self.hidden_layer_size, device=self.device, dtype=self.dtype)
        lstm_out, _ = self.lstm(input_seq, (hs, cs))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.bn_lstm(lstm_out)
        x = self.relu(self.linear1(lstm_out))
        x = self.bn_linear(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        # Reshape back to the batch_size, seq_length, output_size
        x = x.view(input_seq.size(0), self.output_seq_len, -1)
        return x

def split_data_train_test(data: pd.DataFrame, split: float) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    # Split the data into train and test, we want to use sliding window (use the last N days to predict the next day)
    split = int(split * len(data))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[data.columns] = scaler.fit_transform(data)
    train, test = data[:split], data[split:]

    logging.info(f'Train size: {len(train)}')
    logging.info(f'Test size: {len(test)}')

    return train, test, scaler

def dataset_for_timeseries(
        df: pd.DataFrame,
        features: list[str],
        target: str,
        window_size: int,
        prediction_size: int,
        device: torch.device,
        dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    ds = df[features + [target]].values.astype('float32')
    X: list[float] = []
    y: list[float] = []
    for i in range(len(ds) - window_size - prediction_size + 1):
        X.append(ds[i:i + window_size, :-1].tolist())
        y.append(ds[i + window_size:i + window_size + prediction_size, -1:].tolist())
    return (torch.tensor(X, dtype=dtype, device=device),
            torch.tensor(y, dtype=dtype, device=device))

if __name__ == '__main__':
    # ncpu = torch.get_num_threads()
    batch_size = 16
    eval_every_n_batches = 10
    epochs = 2000
    window_size = 5
    prediction_size = 1
    dropout = 0.1
    lstm_layers = 4
    hidden_layer_size = 16
    dense_size = 64
    save_best = True
    plot_history_length = 10
    target = 'COVID-19 admissions (suspected and confirmed)'
    features = [
        target,
        'day_of_week',
        'humidity',
        'temp',
        'windspeed',
        'winddir'
    ]
    train_p = 0.70
    learning_rate = 1e-4
    weight_decay = 1e-5  # L2 regularization
    start_datetime = datetime.timestamp(datetime.now())
    model_base_filename = MODEL_DIR / f'lstm_admissions_model_{start_datetime}'
    metadata_filename = model_base_filename.with_suffix('.json')

    # write metadata
    metadata = {
        'batch_size': batch_size,
        'epochs': epochs,
        'window_size': window_size,
        'prediction_size': prediction_size,
        'dropout': dropout,
        'lstm_layers': lstm_layers,
        'hidden_layer_size': hidden_layer_size,
        'dense_size': dense_size,
        'target': target,
        'features': features,
        'train_p': train_p,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'start_datetime': start_datetime,
        'model_base_filename': str(model_base_filename),
    }
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)



    logging.basicConfig(level=logging.INFO)
    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    # Load the data
    logging.info('Loading data')
    data = read_data()
    data, _ = prepare_data(data)
    
    # only keep target and features
    data = data[features]


    data_size = len(data)
    train, test, scaler = split_data_train_test(data, train_p)
    train_size = len(train)
    test_size = len(test)
    X_train, y_train = dataset_for_timeseries(train, features, target, window_size, prediction_size, device)
    X_test, y_test = dataset_for_timeseries(test, features, target, window_size, prediction_size, device)
    
    logging.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, dtype: {X_train.dtype}')
    logging.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, dtype: {X_test.dtype}')

    model = AdmissionsLSTM(
        input_size = len(features),
        hidden_layer_size = hidden_layer_size,
        lstm_layers = lstm_layers,
        dense_size = dense_size,
        output_seq_len = prediction_size,
        dropout = dropout,
        device = device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    dl = data_utils.DataLoader(data_utils.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)

    early_stopping = EarlyStopping(patience=1000, min_delta=0.0001)

    logging.info('Training LSTM model')
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_combined_loss = float('inf')
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    xticks = list(range(-plot_history_length, 0))
    for i in tqdm.tqdm(range(epochs), colour='red', position=0, leave=True):
        model.train()
        for X_batch, y_batch in dl:
            # logging.info(f'X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}, X_batch dtype: {X_batch.dtype}, y_batch dtype: {y_batch.dtype}')
            # logging.info(f'X_batch device: {X_batch.device}, y_batch device: {y_batch.device}')
            y_pred = model(X_batch)
            single_loss: torch.Tensor = loss_function(y_pred, y_batch)
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()
        if i % eval_every_n_batches == 0:
            # logging.info(f'Epoch: {i}, loss: {single_loss.item()}')
            extra: dict[str, str] = {}
            model.eval()
            with torch.no_grad():
                y_train_pred: torch.Tensor = model(X_train)
                train_loss: float = loss_function(y_train_pred, y_train).item()
                
                train_loss_history.append(train_loss)
                train_rmse = math.sqrt(train_loss)
                # logging.info(f'Train loss: {train_loss}, Train RMSE: {train_rmse}')
                y_test_pred: torch.Tensor = model(X_test)
                test_loss: float = loss_function(y_test_pred, y_test).item()
                test_loss_history.append(test_loss)
                test_rmse = math.sqrt(test_loss)
                combined_loss: float = train_loss + test_loss
                extra['combined loss'] = f'{combined_loss:.4f}'
                if combined_loss < best_combined_loss:
                    best_combined_loss = combined_loss
                    extra['best combined loss'] = f'{best_combined_loss:.4f}*'
                    if save_best:
                        # logging.info('Saving best combined test/train loss model')
                        torch.save(model.state_dict(), model_base_filename.with_suffix('.best_combined.pth'))
                        extra['best combined loss'] += ' [SAVED]'
                else:
                    extra['best combined loss'] = f'{best_combined_loss:.4f}'

                extra['train loss'] = f'{train_loss:.4f}'
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    extra['best train loss'] = f'{best_train_loss:.4f}*'
                    if save_best:
                        # logging.info('Saving best train model')
                        torch.save(model.state_dict(), model_base_filename.with_suffix('.best_train.pth'))
                        extra['best train loss'] += ' [SAVED]'
                else:
                    extra['best train loss'] = f'{best_train_loss:.4f}'
                
                extra['test loss'] = f'{test_loss:.4f}'
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    extra['best test loss'] = f'{best_test_loss:.4f}*'
                    if save_best:
                        # logging.info('Saving best test model')
                        torch.save(model.state_dict(), model_base_filename.with_suffix('.best_test.pth'))
                        extra['best test loss'] += ' [SAVED]'
                else:
                    extra['best test loss'] = f'{best_test_loss:.4f}'

                
                # logging.info(f'Test loss: {test_loss}, Test RMSE: {test_rmse}')
                # plot test and train loss history
                if gnuplot_available:
                    tmpfile = plotting.write_plot_data(list(zip(reversed(xticks), reversed(train_loss_history), reversed(test_loss_history))))
                    plotting.plot_running_loss(
                        gp,
                        tmpfile,
                        extra=extra
                    )
                early_stopping(test_loss)
                if early_stopping.early_stop:
                    logging.info('Early stopping')
                    break

    logging.info('Saving LSTM model')
    torch.save(model.state_dict(), model_base_filename.with_suffix('.last.pth'))
    logging.info('Model saved')

    with torch.no_grad():
        logging.info('Plotting predictions vs actual')
        data_np: np.ndarray = data.to_numpy()
        train_ts = np.ones_like(data_np, dtype='float32') * np.nan
        test_ts = np.ones_like(data_np, dtype='float32') * np.nan

        y_train_pred = model(X_train).cpu().numpy()
        y_test_pred = model(X_test).cpu().numpy()
        logging.info(f'y_train_pred shape: {y_train_pred.shape}, y_test_pred shape: {y_test_pred.shape}')
        train_ts[window_size + (prediction_size - 1):train_size, 0] = y_train_pred[:, -1, -1]
        test_ts[train_size + window_size + (prediction_size - 1):data_size, 0] = y_test_pred[:, -1, -1]
        logging.info(f'Train_ts shape: {train_ts.shape}, Test_ts shape: {test_ts.shape}')
        train_ts = scaler.inverse_transform(train_ts)
        test_ts = scaler.inverse_transform(test_ts)

        data_unscaled = scaler.inverse_transform(data)

        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data_unscaled[:, 0], label='Actual')
        plt.plot(data.index, train_ts[:, 0], label='Train Prediction')
        plt.plot(data.index, test_ts[:, 0], label='Test Prediction')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / 'predictions_vs_actual_lstm.png')
        logging.info('Plot saved')
