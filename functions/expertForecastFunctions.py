import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def base_mapping(financial_df, metrics=None):
    """
    Categorizes financial metrics based on predefined patterns and assigns base metrics for ratio calculations.

    Args:
        financial_df (DataFrame): DataFrame containing financial metrics.
        metrics (dict, optional): Dictionary defining the regex patterns for categorizing metrics.

    Returns:
        tuple: A tuple containing the categorized metrics and the base mappings.
    """
    if metrics is None:
        metrics = {
            "assets": "asset", 
            "liabilities_and_equity": "liabilit|equit|fund", 
            "p_and_l": ""
        }

    if financial_df.index.name != 'Financial Metric':
        financial_df.set_index('Financial Metric', inplace=True)

    categorized_metrics = {}
    for key, regex_pattern in metrics.items():
        if regex_pattern:
            categorized_metrics[key] = [metric for metric in financial_df.index if re.search(regex_pattern, metric, re.IGNORECASE)]
        else:
            already_categorized = {item for sublist in categorized_metrics.values() for item in sublist}
            categorized_metrics[key] = [metric for metric in financial_df.index if metric not in already_categorized]

    base_mapping = {
        "assets": "Total assets",
        "liabilities_and_equity": "Total shareholders' funds and liabilities",
        "p_and_l": "Sales"  # Assuming 'Sales' is the base for p_and_l if not specified.
    }

    return categorized_metrics, base_mapping

def forecast_financials(df, assumptions, base_year, forecast_years):
    """
    Generates financial forecasts based on provided assumptions and historical data.

    Parameters:
        df (DataFrame): Historical financial data.
        assumptions (dict): Growth assumptions for various financial metrics.
        base_year (str): The base year for the forecast (last historical year).
        forecast_years (list): List of years for which the forecast is to be made.
        
    Returns:
        DataFrame: Forecasted financials including CAGR for the forecast period.
    """
    
    forecast_df = pd.DataFrame(index=df.index, columns=forecast_years)
    last_values = df[base_year]

    for year in forecast_years:
        for metric, values in assumptions.items():
            base_value = last_values.get(metric, None)
            if base_value is not None:
                growth_rate = values['rates'][int(''.join(filter(str.isdigit, year))) - int(base_year)]
                forecast_value = base_value * (1 + growth_rate)
                forecast_df.loc[metric, year] = forecast_value
                last_values[metric] = forecast_value

    calculate_cagr(forecast_df, forecast_years)  # Calculate CAGR and add as a column

    return forecast_df

def calculate_cagr(df, forecast_years):
    """
    Calculates the compound annual growth rate (CAGR) for forecasted data.

    Args:
        df (DataFrame): The DataFrame with forecast data.
        forecast_years (list): List of forecast years.
    """
    start_year, end_year = forecast_years[0], forecast_years[-1]
    num_years = len(forecast_years) - 1
    df['CAGR'] = ((df[end_year] / df[start_year]) ** (1 / num_years) - 1).fillna(0) * 100

def handle_missing_values(df, strategy='median'):
    """
    Imputes missing values based on the specified strategy.

    Args:
        df (DataFrame): DataFrame with missing values.
        strategy (str, optional): Strategy to use for imputing missing values. Defaults to 'median'.

    Returns:
        DataFrame: DataFrame with imputed values.
    """
    if strategy == 'model':
        for column in df.columns:
            if df[column].isnull().any():
                impute_with_model(df, column)
    else:
        imputer = SimpleImputer(strategy=strategy)
        df[:] = imputer.fit_transform(df)

    return df

def impute_with_model(df, target_column):
    """
    Imputes missing values using a RandomForestRegressor model for the specified column.

    Args:
        df (DataFrame): DataFrame containing the target column with missing values.
        target_column (str): The column for which the missing values are to be imputed.
    """
    features = df.columns.difference([target_column])
    known_mask = df[target_column].notna()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(df.loc[known_mask, features], df.loc[known_mask, target_column])
    df.loc[~known_mask, target_column] = model.predict(df.loc[~known_mask, features])


def categorize_and_map_metrics(financial_df, metrics_patterns=None):
    """
    Categorizes financial metrics based on provided patterns and assigns base metrics for ratio calculations.
    This supports dynamic relationships based on historical data to improve forecasting accuracy.

    Args:
        financial_df (DataFrame): DataFrame with financial metrics indexed by 'Financial Metric'.
        metrics_patterns (dict, optional): Patterns to categorize metrics. Defaults to a common set.

    Returns:
        tuple: Two dictionaries, one for categorized metrics and another for their base mappings.
    """
    if metrics_patterns is None:
        metrics_patterns = {
            "assets": "asset", 
            "liabilities_and_equity": "liabilit|equit|fund", 
            "p_and_l": "profit|loss|revenue|sales"
        }

    if financial_df.index.name != 'Financial Metric':
        financial_df.set_index('Financial Metric', inplace=True)

    categorized_metrics = {key: [] for key in metrics_patterns.keys()}
    for metric in financial_df.index:
        for category, pattern in metrics_patterns.items():
            if re.search(pattern, metric, re.IGNORECASE):
                categorized_metrics[category].append(metric)
                break

    base_mapping = {}
    for category, metrics in categorized_metrics.items():
        if category == 'assets':
            base_metric = 'Total assets'
        elif category == 'liabilities_and_equity':
            base_metric = 'Total liabilities and equity'
        elif category == 'p_and_l':
            base_metric = 'Total revenue'
        for metric in metrics:
            base_mapping[metric] = base_metric

    return categorized_metrics, base_mapping


def apply_var_model(df, forecast_years):
    if len(df) <= 15:  # Check if there's enough data for the maximum lag
        raise ValueError("Insufficient data for the number of lags considered.")
    if not all(isinstance(year, (int, float)) for year in forecast_years):
        raise ValueError("Forecast years should be numeric.")

    model = VAR(df.dropna())  # Ensure data is complete
    results = model.fit(maxlags=15, ic='aic')
    forecasted_values = results.forecast(df.values[-results.k_ar:], steps=len(forecast_years))
    
    forecast_df = pd.DataFrame(forecasted_values, index=forecast_years, columns=df.columns)
    return forecast_df

def calculate_forecasts(historical_ratios, forecast_years, future_bases, base_mapping):
    forecasts = {}
    projected_ratios = {}
    
    for metric, ratios in historical_ratios.items():
        base = base_mapping[metric]
        if ratios.dropna().empty:
            continue

        # Using Ridge regression with polynomial features for more robust fitting
        model = make_pipeline(PolynomialFeatures(degree=2), Ridge())
        X = np.arange(len(ratios)).reshape(-1, 1)
        y = ratios.values.flatten()

        model.fit(X, y)
        projected = model.predict(np.arange(len(ratios), len(ratios) + len(forecast_years)).reshape(-1, 1))
        
        # Use historical data to estimate variance and define a stability threshold
        historical_variance = np.var(y)
        stability_threshold = np.sqrt(historical_variance) * 2  # 2 standard deviations

        # Applying a stability check before accepting sign changes
        last_valid_value = ratios.iloc[-1]
        corrected_projected = np.where(
            np.abs(projected - last_valid_value) < stability_threshold,
            projected,
            last_valid_value + np.sign(projected - last_valid_value) * stability_threshold
        )

        forecasts[metric] = corrected_projected * future_bases[base].loc[forecast_years].values
        projected_ratios[metric] = corrected_projected

    forecast_df = pd.DataFrame(forecasts, index=forecast_years)
    projected_ratios_df = pd.DataFrame(projected_ratios, index=forecast_years) * 100
    
    return forecast_df, projected_ratios_df

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

    return result_forecast, result_percentages
