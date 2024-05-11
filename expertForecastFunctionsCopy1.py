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
                growth_rate = values['rates'][int(''.join(filter(str.isdigit, year))) - int(base_year) - 1]
                forecast_value = base_value * (1 + growth_rate)
                forecast_df.loc[metric, year] = forecast_value
                last_values[metric] = forecast_value

    calculate_cagr(forecast_df, base_year, forecast_years)  # Calculate CAGR and add as a column

    return forecast_df


import pandas as pd

def rolling_forecast_financials(df, assumptions, base_year, forecast_years, cagr_period=5):
    """
    Generates financial forecasts based on provided assumptions, historical data,
    and rolling CAGR calculations that span both historical and forecasted data.

    Parameters:
        df (DataFrame): Historical financial data up to the base year.
        assumptions (dict): Explicit growth assumptions for various financial metrics.
        base_year (str): The base year for the forecast (last historical year).
        forecast_years (list): List of years for which the forecast is to be made.
        cagr_period (int): The number of years to use for rolling CAGR calculation.
        
    Returns:
        DataFrame: Forecasted financials including dynamic CAGR for the forecast period.
    """
    
    forecast_df = pd.DataFrame(index=df.index, columns=forecast_years)
    last_values = df[base_year].copy()

    # Initialize combined_df with historical data up to the base year
    combined_df = df.copy()

    for year_index, year in enumerate(forecast_years):
        current_forecast_year = year.rstrip('F')

        # Calculate forecast values for the current year
        for metric in df.index:
            # Check if there are enough assumptions rates for the current year
            if metric in assumptions and year_index < len(assumptions[metric]['rates']):
                # Use explicit growth rate if available
                growth_rate = assumptions[metric]['rates'][year_index]
            else:
                # Determine the start and end years for CAGR calculation
                end_year = int(current_forecast_year) - 1  # End year is the year before the current forecast year
                start_year = max(end_year - cagr_period + 1, int(base_year) - cagr_period + 1)

                # Calculate rolling CAGR using combined data
                start_value = combined_df.loc[metric, str(start_year)]
                end_value = combined_df.loc[metric, str(end_year)]
                if start_value != 0:
                    growth_rate = (end_value / start_value) ** (1 / (end_year - start_year)) - 1
                else:
                    growth_rate = 0

            # Calculate forecast value
            base_value = last_values.get(metric, 0)
            forecast_value = base_value * (1 + growth_rate)
            forecast_df.loc[metric, year] = forecast_value
            last_values[metric] = forecast_value

        # Update combined_df with the forecast for the current year
        combined_df[current_forecast_year] = forecast_df[year]

    # Calculate CAGR for the entire forecast period and add as a column
    calculate_cagr(combined_df, forecast_df, base_year, forecast_years)

    return forecast_df

def calculate_total_cagr(df, forecast_df, base_year, forecast_years):
    """
    Calculates the compound annual growth rate (CAGR) for forecasted data.

    Args:
        df (DataFrame): The DataFrame with combined historical and forecast data.
        forecast_df (DataFrame): The DataFrame with forecast data.
        base_year (str): The starting year for historical data.
        forecast_years (list): List of forecast years.
    """
    start_year = base_year
    end_year = forecast_years[-1]
    num_years = len(forecast_years)
    forecast_df['CAGR'] = ((forecast_df[end_year] / df[start_year]) ** (1 / num_years) - 1).fillna(0) * 100


def calculate_cagr(df, base_year, forecast_years):
    """
    Calculates the compound annual growth rate (CAGR) for forecasted data.

    Args:
        df (DataFrame): The DataFrame with forecast data.
        forecast_years (list): List of forecast years.
    """
    start_year, end_year = base_year, forecast_years[-1]
    num_years = len(forecast_years)
    df['CAGR'] = ((df[end_year] / df[start_year]) ** (1 / num_years) - 1).fillna(0) * 100

def handle_missing_values(df, model_cols=None):
    if model_cols is None:
        model_cols = []

    if 'Costs of employees' not in df.columns:
        df = df.T

    cols_with_missing = df.columns[df.isnull().any()].tolist()
    imputation_strategies = {
        col: ('model' if df[col].dtype.kind in 'biufc' else 'most_frequent')
        for col in cols_with_missing
    }
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
    
    if 'Costs of employees' in df.columns:
        df = df.T

    return df

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
