import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def base_mapping(financial_df, metrics=None):
    if metrics is None:
        metrics = {
            "assets": "asset", 
            "liabilities_and_equity": "liabilit|equit|fund", 
            "p_and_l": ""
        }

    if financial_df.index.name != 'Financial Metric':
        financial_df.set_index('Financial Metric', inplace=True)

    categorized_metrics = {}
    for key, maps in metrics.items():
        if maps:
            categorized_metrics[key] = [metric for metric in financial_df.index if re.search(maps, metric, re.IGNORECASE)]
        else:
            already_categorized = {item for sublist in categorized_metrics.values() for item in sublist}
            categorized_metrics[key] = [metric for metric in financial_df.index if metric not in already_categorized]

    base_mapping = {}
    for item in categorized_metrics['assets']:
        base_mapping[item] = 'Total assets'
    for item in categorized_metrics['liabilities_and_equity']:
        base_mapping[item] = 'Total shareholders\' funds and liabilities'
    for item in categorized_metrics['p_and_l']:
        base_mapping[item] = 'Sales' if 'Revenue' in item or 'Sales' in item else 'Sales'

    return categorized_metrics, base_mapping

def forecast_financials(df, assumptions, base_year, forecast_years):
    yoy_forecast_df = pd.DataFrame(index=df.index, columns=forecast_years)
    for idx, year in enumerate(forecast_years):
        for key, value in assumptions.items():
            if idx == 0:
                last_value = df.loc[df.index.str.contains(key), base_year].values[0]
            growth_rate = value['rates'][idx]
            forecast_value = last_value * (1 + growth_rate)
            yoy_forecast_df.loc[key, year] = forecast_value
            last_value = forecast_value

    yoy_forecast_df /= 1e3
    yoy_cagr_df = ((yoy_forecast_df[forecast_years].astype(float).iloc[:, -1] /
                    yoy_forecast_df[forecast_years].astype(float).iloc[:, 0]) ** (1 / (len(forecast_years) - 1)) - 1) * 100
    yoy_forecast_df.loc[:, 'CAGR'] = yoy_cagr_df
    return yoy_forecast_df

def handle_missing_values(df, model_cols=None):
    if model_cols is None:
        model_cols = []

    if 'Costs of employees' not in df.columns:
        df = df.T

    cols_with_missing = df.columns[df.isnull().any()].tolist()
    imputation_strategies = {col: ('model' if col in model_cols else 'most_frequent') for col in cols_with_missing}

    for col, strategy in imputation_strategies.items():
        if strategy != 'model':
            imputer = SimpleImputer(strategy=strategy)
            df[col] = imputer.fit_transform(df[[col]])
        else:
            features = df.columns.difference([col] + cols_with_missing).tolist()
            train_data = df.dropna(subset=[col] + features)
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_data[features])
            model = RandomForestRegressor(random_state=0)
            model.fit(train_features_scaled, train_data[col])
            test_features = scaler.transform(df.loc[df[col].isnull(), features])
            df.loc[df[col].isnull(), col] = model.predict(test_features)

    return df

def linear_regression_forecast(df, forecast_years, transform_if_needed=True):
    if transform_if_needed and '2021' not in df.columns:
        df = df.T

    historical_years = df.columns[df.columns.str.isnumeric()]
    historical_regression_forecast = df.apply(
        lambda x: forecast_metric(x, historical_years, forecast_years), axis=1) / 1e3

    historical_cagr_values = ((historical_regression_forecast[forecast_years].astype(float).iloc[:, -1] /
                               historical_regression_forecast[forecast_years].astype(float).iloc[:, 0]) ** 
                              (1 / (len(forecast_years) - 1)) - 1) * 100
    historical_regression_forecast.loc[:, 'CAGR'] = historical_cagr_values
    return historical_regression_forecast

def ratio_forecast_regression(df, historical_regression_forecast, base_mapping, forecast_years):
    future_bases = {base: historical_regression_forecast.loc[base] for base in set(base_mapping.values())}
    historical_ratios = {metric: df.loc[metric] / df.loc[base] for metric, base in base_mapping.items()}
    historical_ratios = pd.DataFrame.from_dict(historical_ratios).dropna(axis='columns')
    forecasts, projected_ratios = calculate_forecasts(historical_ratios, forecast_years, future_bases, base_mapping)

    result_forecast = pd.DataFrame([{'Financial Metric': metric, 'Year': year, 'Value': value}
                                    for (metric, year), value in forecasts.items()]).pivot(
        index='Financial Metric', columns='Year', values='Value')
    result_percentages = pd.DataFrame([{'Financial Metric': metric, 'Year': year, 'Value': value}
                                       for (metric, year), value in projected_ratios.items()]).pivot(
        index='Financial Metric', columns='Year', values='Value') * 100
    result_forecast.loc[:, 'CAGR'] = calculate_cagr(result_forecast, forecast_years)
    return result_forecast, result_percentages
