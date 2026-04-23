import argparse
import math
from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor


def load_or_generate_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        if 'date' not in df.columns or 'consumption' not in df.columns:
            raise ValueError("CSV должен содержать столбцы: date, consumption")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    # Синтетический пример на 365 дней, если пользователь ещё не подготовил CSV
    rng = np.random.default_rng(42)
    dates = pd.date_range('2025-01-01', periods=365, freq='D')
    t = np.arange(365)
    seasonal_week = 15 * np.sin(2 * np.pi * t / 7)
    seasonal_year = 40 * np.sin(2 * np.pi * t / 365)
    trend = 0.08 * t
    noise = rng.normal(0, 4, size=365)
    consumption = 220 + seasonal_week + seasonal_year + trend + noise
    return pd.DataFrame({'date': dates, 'consumption': consumption.round(2)})


def minmax_scale(values: np.ndarray):
    x_min = values.min()
    x_max = values.max()
    if math.isclose(float(x_max), float(x_min)):
        scaled = np.zeros_like(values, dtype=float)
    else:
        scaled = (values - x_min) / (x_max - x_min)
    return scaled, float(x_min), float(x_max)


def inverse_scale(values: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return values * (x_max - x_min) + x_min


def make_windows(series: np.ndarray, window: int = 7):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100


def train_and_forecast(df: pd.DataFrame, out_dir: Path, window: int = 7):
    out_dir.mkdir(parents=True, exist_ok=True)

    values = df['consumption'].to_numpy(dtype=float)
    scaled, x_min, x_max = minmax_scale(values)
    X, y = make_windows(scaled, window=window)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = MLPRegressor(
        hidden_layer_sizes=(15,),
        activation='logistic',
        solver='sgd',
        learning_rate_init=0.3,
        momentum=0.9,
        max_iter=500,
        tol=0.01,
        random_state=42,
        n_iter_no_change=30,
    )
    model.fit(X_train, y_train)

    train_loss_curve = getattr(model, 'loss_curve_', [])

    y_test_pred_scaled = model.predict(X_test)
    y_test_true = inverse_scale(y_test, x_min, x_max)
    y_test_pred = inverse_scale(y_test_pred_scaled, x_min, x_max)

    mse = mean_squared_error(y_test_true, y_test_pred)
    mae = mean_absolute_error(y_test_true, y_test_pred)
    mape_value = mape(y_test_true, y_test_pred)

    # Итеративный прогноз на 7 дней вперёд
    last_window = scaled[-window:].copy()
    future_scaled = []
    for _ in range(7):
        next_scaled = model.predict(last_window.reshape(1, -1))[0]
        future_scaled.append(next_scaled)
        last_window = np.roll(last_window, -1)
        last_window[-1] = next_scaled

    future_values = inverse_scale(np.array(future_scaled), x_min, x_max)
    start_date = df['date'].iloc[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start_date, periods=7, freq='D')

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast_consumption': np.round(future_values, 2)
    })

    # Таблица сравнения test
    test_dates = df['date'].iloc[window + split: window + split + len(y_test_true)].reset_index(drop=True)
    compare_df = pd.DataFrame({
        'date': test_dates,
        'real_consumption': np.round(y_test_true, 2),
        'predicted_consumption': np.round(y_test_pred, 2),
        'abs_error': np.round(np.abs(y_test_true - y_test_pred), 2)
    })

    metrics_df = pd.DataFrame({
        'metric': ['MSE', 'MAE', 'MAPE_%'],
        'value': [round(float(mse), 4), round(float(mae), 4), round(float(mape_value), 4)]
    })

    # Сохранение артефактов
    data_path = out_dir / 'data_used.csv'
    df.to_csv(data_path, index=False)
    forecast_df.to_csv(out_dir / 'forecast_next_7_days.csv', index=False)
    compare_df.to_csv(out_dir / 'test_predictions.csv', index=False)
    metrics_df.to_csv(out_dir / 'metrics.csv', index=False)

    with open(out_dir / 'energy_forecast_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'x_min': x_min,
            'x_max': x_max,
            'window': window,
        }, f)

    # График реальные vs прогнозируемые
    plt.figure(figsize=(11, 5))
    plt.plot(df['date'], df['consumption'], label='Реальные исторические данные')
    plt.plot(compare_df['date'], compare_df['predicted_consumption'], label='Прогноз на тесте')
    plt.plot(forecast_df['date'], forecast_df['forecast_consumption'], label='Прогноз на 7 дней вперёд')
    plt.xlabel('Дата')
    plt.ylabel('Потребление электроэнергии')
    plt.title('Прогнозирование временного ряда с помощью MLP (Python)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'forecast_plot.png', dpi=150)
    plt.close()

    # График ошибки обучения
    if train_loss_curve:
        plt.figure(figsize=(9, 4.5))
        plt.plot(range(1, len(train_loss_curve) + 1), train_loss_curve)
        plt.xlabel('Эпоха')
        plt.ylabel('Ошибка')
        plt.title('График ошибки обучения')
        plt.tight_layout()
        plt.savefig(out_dir / 'training_error_plot.png', dpi=150)
        plt.close()

    return {
        'metrics': metrics_df,
        'forecast': forecast_df,
        'compare': compare_df,
        'data_path': data_path,
    }


def save_template_csv(path: Path):
    dates = pd.date_range('2025-01-01', periods=14, freq='D')
    consumption = [210, 215, 208, 220, 225, 230, 227, 219, 221, 224, 228, 232, 229, 231]
    pd.DataFrame({'date': dates, 'consumption': consumption}).to_csv(path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Практическая работа №8: прогнозирование временных рядов на Python')
    parser.add_argument('--input', default='energy_data.csv', help='CSV-файл с колонками date,consumption')
    parser.add_argument('--output', default='results_pr8', help='Папка для результатов')
    parser.add_argument('--make-template', action='store_true', help='Создать шаблон CSV и выйти')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.make_template:
        save_template_csv(input_path)
        print(f'Шаблон сохранён: {input_path}')
        raise SystemExit(0)

    df = load_or_generate_data(input_path)
    results = train_and_forecast(df, output_path)

    print('Готово. Результаты сохранены в:', output_path)
    print('\nМетрики качества:')
    print(results['metrics'].to_string(index=False))
    print('\nПрогноз на 7 дней:')
    print(results['forecast'].to_string(index=False))
