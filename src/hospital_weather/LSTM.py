import json
import logging
import math
from typing import Any
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps
from .common.config import read_data, prepare_data, MODEL_DIR, SELECTED_FEATURES, SELECTED_TARGET
from .common import plotting
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler  # type: ignore
import sklearn.metrics  # type: ignore
from datetime import datetime
import matplotlib.dates as mdates


# https://stackoverflow.com/a/49982967
def reverse_stripe(a: np.ndarray) -> np.ndarray:
    a = np.asanyarray(a)
    *sh, i, j = a.shape
    assert i >= j
    *st, k, m = a.strides
    return np.lib.stride_tricks.as_strided(a[..., j-1:, :], (*sh, i-j+1, j), (*st, k, m-k))
        


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
            lstm_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = False,
            device: torch.device = torch.device('cpu'),
            dtype=torch.float32):
        super().__init__()
        self.device = device
        self.lstm_layers = lstm_layers
        self.hidden_layer_size = hidden_layer_size
        self.output_seq_len = output_seq_len
        self.dtype = dtype
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            device=device,
            dtype=dtype
        )
        # self.bn_lstm = nn.BatchNorm1d(hidden_layer_size, device=device, dtype=dtype)
        self.linear1 = nn.Linear(hidden_layer_size * self.num_directions, dense_size, device=device, dtype=dtype)
        # self.bn_linear = nn.BatchNorm1d(dense_size, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dense_size, output_size * output_seq_len, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_seq):
        hs = torch.zeros(self.lstm_layers  * self.num_directions, input_seq.size(0), self.hidden_layer_size, device=self.device, dtype=self.dtype)
        cs = torch.zeros(self.lstm_layers  * self.num_directions, input_seq.size(0), self.hidden_layer_size, device=self.device, dtype=self.dtype)
        lstm_out, _ = self.lstm(input_seq, (hs, cs))
        lstm_out = lstm_out[:, -1, :]
        
        # lstm_out = self.bn_lstm(lstm_out)

        x = self.relu(self.linear1(lstm_out))
        # x = self.bn_linear(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        # Reshape back to the batch_size, seq_length, output_size
        x = x.view(input_seq.size(0), self.output_seq_len, -1)
        return x

def split_data_train_test(data: pd.DataFrame, split: float, window_size: int, scale_target: bool) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    # Split the data into train and test, we want to use sliding window (use the last N days to predict the next day)
    split = int(split * len(data))
    scaler = MinMaxScaler(feature_range=(0, 1))
    if scale_target:
        data[data.columns] = scaler.fit_transform(data)
    else:
        data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
    train, test = data[:split], data[split-window_size:]

    logging.info(f'Train size: {len(train)}')
    logging.info(f'Test size: {len(test)}')

    return train, test, scaler

def dataset_for_timeseries(
        df: pd.DataFrame,
        features: list[str],
        window_size: int,
        prediction_size: int,
        device: torch.device,
        dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    ds = df[features].values
    X: list[float] = []
    y: list[float] = []
    for i in range(len(ds) - window_size - prediction_size):
        X.append(ds[i:i + window_size, :].tolist())
        y.append(ds[i + window_size:i + window_size + prediction_size, -1:].tolist())
    return (torch.tensor(X, dtype=dtype, device=device),
            torch.tensor(y, dtype=dtype, device=device))

def train_lstm() -> None:
    # ncpu = torch.get_num_threads()
    batch_size = 64
    eval_every_n_epochs = 10
    epochs = 10000
    window_size = 14
    prediction_size = 3
    dropout = 0.1
    lstm_layers = 3 # underfits with 2
    hidden_layer_size = 250
    dense_size = 128
    save_best = True
    plot_history_length = 10
    early_stopping_patience = 500
    early_stopping_min_delta = 1e-6
    scale_target = True
    bidirectional = True
    target = SELECTED_TARGET
    features = SELECTED_FEATURES
    train_p = 0.70
    learning_rate = 1e-4
    weight_decay = 1e-5  # L2 regularization
    start_datetime = datetime.timestamp(datetime.now())
    model_base_filename = MODEL_DIR / f'lstm_admissions_model_{start_datetime}'
    metadata_filename = model_base_filename.with_suffix('.json')


    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

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
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_min_delta': early_stopping_min_delta,
        'scale_target': scale_target,
        'bidirectional': bidirectional,
    }
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)



    logging.basicConfig(level=logging.INFO)
    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    # Load the data
    logging.info('Loading data')
    data = prepare_data(read_data())
    
    # only keep target and features
    data = data[features]

    logging.info(f'Data shape: {data.shape}')
    # make sure all values are float32
    data = data.astype('float32')
    
    train, test, scaler = split_data_train_test(data, train_p, window_size, scale_target=scale_target)

    X_train, y_train = dataset_for_timeseries(train, features, window_size, prediction_size, device)
    X_test, y_test = dataset_for_timeseries(test, features, window_size, prediction_size, device)
    
    logging.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, dtype: {X_train.dtype}')
    logging.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, dtype: {X_test.dtype}')

    model = AdmissionsLSTM(
        input_size = len(features),
        hidden_layer_size = hidden_layer_size,
        lstm_layers = lstm_layers,
        dense_size = dense_size,
        output_seq_len = prediction_size,
        dropout = dropout,
        bidirectional = bidirectional,
        device = device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, min_lr=1e-5)
    dl = data_utils.DataLoader(data_utils.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)

    # # load MODEL_DIR lstm_admissions_model_1717106911.best_combined.pth
    # model.load_state_dict(torch.load(MODEL_DIR / 'lstm_admissions_model_1717106911.best_combined.pth'))

    # # export to onnx
    # input_sample = torch.randn(1, window_size, len(features), device=device)
    # torch.onnx.export(model, input_sample, str(MODEL_DIR / "lstm_admissions_model_1717106911.best_combined.onnx"), verbose=True)
    # exit()


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
            optimizer.zero_grad()
            y_pred = model(X_batch)
            single_loss: torch.Tensor = loss_function(y_pred, y_batch)
            single_loss.backward()
            optimizer.step()
        if i % eval_every_n_epochs == 0:
            extra: dict[str, str] = {}
            model.eval()
            with torch.no_grad():
                y_train_pred: torch.Tensor = model(X_train)
                train_loss: float = math.sqrt(loss_function(y_train_pred, y_train).item())
                train_loss_history.append(train_loss)

                y_test_pred: torch.Tensor = model(X_test)
                test_loss: float = math.sqrt(loss_function(y_test_pred, y_test).item())
                test_loss_history.append(test_loss)

                combined_loss: float = math.sqrt(train_loss**2 + test_loss**2)
                extra['rmse (train+test)'] = f'{combined_loss:.4f}'
                if combined_loss < best_combined_loss:
                    best_combined_loss = combined_loss
                    extra['best rmse (train+test)'] = f'{best_combined_loss:.4f}*'
                    if save_best:
                        # logging.info('Saving best combined test/train loss model')
                        torch.save(model.state_dict(), model_base_filename.with_suffix('.best_combined.pth'))
                        extra['best rmse (train+test)'] += ' [SAVED]'
                else:
                    extra['best rmse (train+test)'] = f'{best_combined_loss:.4f}'

                extra['train rmse'] = f'{train_loss:.4f}'
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    extra['best train rmse'] = f'{best_train_loss:.4f}*'
                    if save_best:
                        # logging.info('Saving best train model')
                        torch.save(model.state_dict(), model_base_filename.with_suffix('.best_train.pth'))
                        extra['best train rmse'] += ' [SAVED]'
                else:
                    extra['best train rmse'] = f'{best_train_loss:.4f}'
                
                extra['test rmse'] = f'{test_loss:.4f}'
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    extra['best test rmse'] = f'{best_test_loss:.4f}*'
                    if save_best:
                        # logging.info('Saving best test model')
                        torch.save(model.state_dict(), model_base_filename.with_suffix('.best_test.pth'))
                        extra['best test rmse'] += ' [SAVED]'
                else:
                    extra['best test rmse'] = f'{best_test_loss:.4f}'

                scheduler.step(test_loss)

                extra['lr'] = f'{scheduler.get_last_lr()[0]:.2e}'

                # live plot
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

    logging.info('Loading best (test) model')
    model.load_state_dict(torch.load(model_base_filename.with_suffix('.best_test.pth')))

    with torch.no_grad():
        logging.info('Plotting predictions vs actual')
        real_values = data[target].to_numpy()
        train_pred: np.ndarray = model(X_train).cpu().numpy()[:, :, -1]
        test_pred: np.ndarray = model(X_test).cpu().numpy()[:, :, -1]
        if scale_target:
            # manually inverse transform
            target_index = data.columns.get_loc(target)
            train_pred = train_pred * (1/scaler.scale_[target_index]) + scaler.min_[target_index]
            test_pred = test_pred * (1/scaler.scale_[target_index]) + scaler.min_[target_index]
            real_values = real_values * (1/scaler.scale_[target_index]) + scaler.min_[target_index] 


        

        # get quantiles for stripes        
        train_pred_q = np.quantile(reverse_stripe(train_pred), [0.1, 0.5, 0.9], axis=1)
        test_pred_q = np.quantile(reverse_stripe(test_pred), [0.1, 0.5, 0.9], axis=1)

        corner = (prediction_size-1)
        plt.figure(figsize=(20, 20))
        plt.plot(data.index, real_values, label='Actual', color='gray')

        train_start_idx = window_size+corner
        train_end_idx = train_start_idx + train_pred_q.shape[1]
        print(train_start_idx, train_end_idx)
        plt.plot(data.index[train_start_idx:train_end_idx], train_pred_q[1], label='Train Prediction')
        plt.fill_between(data.index[train_start_idx:train_end_idx], train_pred_q[0], train_pred_q[2], alpha=0.5)

        test_start_idx = train_end_idx+corner
        test_end_idx = test_start_idx + test_pred_q.shape[1]
        plt.plot(data.index[test_start_idx:test_end_idx], test_pred_q[1], label='Test Prediction')
        plt.fill_between(data.index[test_start_idx:test_end_idx], test_pred_q[0], test_pred_q[2], alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_base_filename.with_suffix('.png'), dpi=300, bbox_inches='tight')
        logging.info('Plot saved')
        


def compare_lstm_models() -> None:
    logging.basicConfig(level=logging.INFO)
    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the data
    data = prepare_data(read_data())
    data = data[['cov19', 'precip_sum', 'pressure_mean', 'pressure_std', 'temp_mean', 'temp_std', 'windspeed_mean', 'windspeed_std', 'winddir_sin', 'winddir_cos', 'snowdepth_max']]
    data = data.astype('float32')
    logging.info(f'Minimum cov19: {data["cov19"].min()}')
    logging.info(f'Maximum cov19: {data["cov19"].max()}')
    # split the data
    train_p = 0.70
    window_size = 14
    prediction_size = 3
    target = 'cov19'
    features = [
        'precip_sum', 
        'pressure_mean',
        'pressure_std', 
        'temp_mean',
        'temp_std',
        'windspeed_mean',
        'windspeed_std',
        'winddir_sin',
        'winddir_cos',
        'snowdepth_max',
        target,
    ]
    train, test, scaler = split_data_train_test(data, train_p, window_size, scale_target=True)
    X_train, y_train = dataset_for_timeseries(train, features, window_size, prediction_size, device)
    X_test, y_test = dataset_for_timeseries(test, features, window_size, prediction_size, device)
    y_train = y_train[:, :, -1]
    y_test = y_test[:, :, -1]
    # inverse transform
    target_index = data.columns.get_loc(target)
    logging.info(f'Columns: {data.columns}')
    logging.info(f'Target index: {target_index}')
    logging.info(f'Scaler scale: {scaler.scale_}')
    logging.info(f'Scaler min: {scaler.min_}')
    y_train = y_train * (1/scaler.scale_[target_index]) + scaler.min_[target_index]
    y_test = y_test * (1/scaler.scale_[target_index]) + scaler.min_[target_index]
    y_true = data[target]  * (1/scaler.scale_[target_index]) + scaler.min_[target_index]
    model_results: list[dict[str, Any]] = []
    # plot the predictions
    with torch.no_grad():
        logging.info('Calculating model prediction stats')

        for model_file in list(MODEL_DIR.glob('lstm_admissions_model_*.pth')):
            # load model JSON (it is the same name as the model file, but without the .best_xxx.pth)
            # remove the .best_{combined|test|train}.pth suffix
            logging.info(f'Loading model: {model_file}')
            metadata_filename = model_file.with_name(model_file.name.split(".", 1)[0] + '.json')


            metadata: dict[str, Any]
            with open(metadata_filename) as f:
                metadata = json.load(f)
            model = AdmissionsLSTM(
                input_size = len(features),
                hidden_layer_size = metadata['hidden_layer_size'],
                lstm_layers = metadata['lstm_layers'],
                dense_size = metadata['dense_size'],
                output_seq_len = prediction_size,
                dropout = metadata['dropout'],
                device = device,
            )

            model.load_state_dict(torch.load(model_file))
            model.to(device)
            model.eval()

            train_pred: np.ndarray = model(X_train).cpu().numpy()[:, :, -1]
            test_pred: np.ndarray = model(X_test).cpu().numpy()[:, :, -1]

            if metadata['scale_target']:
                # inverse transform
                y_train_pred = train_pred * (1/scaler.scale_[target_index]) + scaler.min_[target_index]
                y_test_pred = test_pred * (1/scaler.scale_[target_index]) + scaler.min_[target_index]
            else:
                y_train_pred = train_pred
                y_test_pred = test_pred

            # show a few example predictions vs actual values
            logging.info('Example predictions vs actual values')
            for i in range(5):
                logging.info(f'Train: {y_train_pred[i]} vs {y_train[i]}')
                logging.info(f'Test: {y_test_pred[i]} vs {y_test[i]}')
            multioutput='uniform_average'

            mae_train = sklearn.metrics.mean_absolute_error(y_train_pred, y_train.cpu().numpy(), multioutput=multioutput)
            mae_test = sklearn.metrics.mean_absolute_error(y_test_pred, y_test.cpu().numpy(), multioutput=multioutput)
            rmse_train = sklearn.metrics.root_mean_squared_error(y_train_pred, y_train.cpu().numpy(), multioutput=multioutput)  # type: ignore
            rmse_test = sklearn.metrics.root_mean_squared_error(y_test_pred, y_test.cpu().numpy(), multioutput=multioutput)  # type: ignore
            r2_train = sklearn.metrics.r2_score(y_train_pred, y_train.cpu().numpy(), multioutput=multioutput)
            r2_test = sklearn.metrics.r2_score(y_test_pred, y_test.cpu().numpy(), multioutput=multioutput)
            mape_train = sklearn.metrics.mean_absolute_percentage_error(y_train_pred, y_train.cpu().numpy(), multioutput=multioutput)
            mape_test = sklearn.metrics.mean_absolute_percentage_error(y_test_pred, y_test.cpu().numpy(), multioutput=multioutput)
            medae_train = sklearn.metrics.median_absolute_error(y_train_pred, y_train.cpu().numpy(), multioutput=multioutput)
            medae_test = sklearn.metrics.median_absolute_error(y_test_pred, y_test.cpu().numpy(), multioutput=multioutput)
            ev_train = sklearn.metrics.explained_variance_score(y_train_pred, y_train.cpu().numpy(), multioutput=multioutput)
            ev_test = sklearn.metrics.explained_variance_score(y_test_pred, y_test.cpu().numpy(), multioutput=multioutput)

            model_name = model_file.name.split('.')[-3].split('_')[-1]
            model_type = model_file.name.split('.')[-2]
            model_results.append({
                'model': 'LSTM_'+model_name,
                'type': model_type,
                'mae_train': float(mae_train),
                'mae_test': float(mae_test),
                'rmse_train': float(rmse_train),
                'rmse_test': float(rmse_test),
                'r2_train': float(r2_train),
                'r2_test': float(r2_test),
                'mape_train': float(mape_train),
                'mape_test': float(mape_test),
                'medae_train': float(medae_train),
                'medae_test': float(medae_test),
                'ev_train': float(ev_train),
                'ev_test': float(ev_test),
            })
            
            # create a plot of the predictions
            if model_type == 'best_test':
                logging.info(f'Plotting predictions for {model_type} model')
                logging.info(y_train.shape)
                logging.info(y_test.shape)
                logging.info(data.index.shape)
                last_train_preds = np.zeros((prediction_size, prediction_size))
                last_test_preds = np.zeros((prediction_size, prediction_size))
                for i in range(prediction_size):
                    last_train_preds[i, :] = np.repeat(y_train_pred[-1, i], prediction_size)
                    last_test_preds[i, :] = np.repeat(y_test_pred[-1, i], prediction_size)
                
                y_train_pred = np.concatenate([y_train_pred, last_train_preds, np.repeat(last_train_preds[-1], prediction_size-1).reshape(prediction_size-1, prediction_size)])
                y_test_pred = np.concatenate([y_test_pred, last_test_preds, np.repeat(last_test_preds[-1], prediction_size-1).reshape(prediction_size-1, prediction_size)])
                logging.info(y_train_pred.shape)
                logging.info(y_test_pred.shape)
                train_pred_q = np.quantile(reverse_stripe(y_train_pred), [0.1, 0.5, 0.9], axis=1)
                test_pred_q = np.quantile(reverse_stripe(y_test_pred), [0.1, 0.5, 0.9], axis=1)
                logging.info(train_pred_q.shape)
                logging.info(test_pred_q.shape)

                start_cutoff = -2*test_pred_q.shape[1]
                plt.figure(figsize=(20, 15))
                plt.rcParams.update({'font.size': 15})
                date_formater = mdates.DateFormatter('%b, %Y')
                plt.scatter(data.index[start_cutoff:], y_true[start_cutoff:], label='Actual', color='gray')

                plt.plot(data.index[start_cutoff:-(test_pred_q.shape[1])], train_pred_q[1, -test_pred_q.shape[1]:], label='Train Prediction')
                plt.fill_between(data.index[start_cutoff:-(test_pred_q.shape[1])], train_pred_q[0, -test_pred_q.shape[1]:], train_pred_q[2, -test_pred_q.shape[1]:], alpha=0.5)

                plt.plot(data.index[start_cutoff+len(train_pred_q[1, -test_pred_q.shape[1]:]):], test_pred_q[1], label='Test Prediction')
                plt.fill_between(data.index[start_cutoff+len(train_pred_q[1, -test_pred_q.shape[1]:]):], test_pred_q[0], test_pred_q[2], alpha=0.5)

                # set date formatter
                plt.gca().xaxis.set_major_formatter(date_formater)
                plt.title(f'LSTM {model_name} Predictions')
                plt.legend()
                plt.tight_layout()
                plt.savefig(MODEL_DIR / f'lstm_admissions_model_comparison_{model_name}.png', dpi=300, bbox_inches='tight')

                # if model_name == '1722541047':
                #     # save onnx
                #     input_sample = torch.randn(1, window_size, len(features), device=device)
                #     torch.onnx.export(model, input_sample, str(MODEL_DIR / f"lstm_admissions_model_{model_name}.onnx"), verbose=True)

        
        
        results = pd.DataFrame(model_results)
        # save results to csv
        logging.info('Saving model comparison results')
        results.to_csv(MODEL_DIR / 'lstm_admissions_model_comparison_results.csv', index=False)
        # keep only "best_test" type
        results = results[results['type'] == 'best_test']
        # rename model from filename
        # model_names = {result['model']: result['model'].split('.')[-3].split('_')[-1] + ' ' + result['model'].split('.')[-2] for result in model_results}
        # plot the results
        logging.info('Plotting model evaluation results')
        plt.figure(figsize=(20, 15))
        plt.rcParams.update({'font.size': 15})
        # plot each of the metrics, one subplot for each, model is X axis, metric score is Y axis, color by type
        cmap = colormaps.get_cmap('rainbow')(np.linspace(0, 1, len(results['type'].unique())))
        colors = {t: c for t, c in zip(results['type'].unique(), cmap)}
        if len(results['type'].unique()) > 1:
            c = results['type'].map(colors)
        else:
            c = None

        metrics =[
            ('mae', 'Mean Absolute Error'),
            ('rmse', 'Root Mean Squared Error'),
            ('r2', 'R2'),
            ('mape', 'Mean Absolute Percentage Error'),
            ('medae', 'Median Absolute Error'),
            ('ev', 'Explained Variance'),
        ]
        for idx, (metric, title) in enumerate(metrics):
            
            ax = plt.subplot(3, 2, idx + 1)
            ax.scatter(results['model'], results[f'{metric}_train'], label='Train', marker='o', alpha=0.5, c=c)
            ax.scatter(results['model'], results[f'{metric}_test'], label='Test', marker='x', c=c)
            # add label for best and worst test score
            best_test = results.loc[results[f'{metric}_test'].idxmin()]
            worst_test = results.loc[results[f'{metric}_test'].idxmax()]
            ax.text(best_test['model'], best_test[f'{metric}_test'], f'Min: {best_test[f"{metric}_test"]:.8f}', fontsize=8, ha='center', va='bottom')  # type: ignore
            ax.text(worst_test['model'], worst_test[f'{metric}_test'], f'Max: {worst_test[f"{metric}_test"]:.8f}', fontsize=8, ha='center', va='top')  # type: ignore
            ax.set_title(title)
            # rotate x labels
            plt.xticks(rotation=60)
            # make x label font smaller
            plt.setp(ax.get_xticklabels(), fontsize=8)
            ax.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / 'lstm_admissions_model_comparison_results.png', dpi=300, bbox_inches='tight')
        
