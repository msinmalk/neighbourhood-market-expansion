import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit

class DataProcessor:
    def __init__(self, walmart_path):
        self.walmart_path = walmart_path

    def load_excel_data(self, file_name, sheet_name=None, skip_rows=0):
        path = self.walmart_path + file_name
        if sheet_name:
            return pd.read_excel(path, sheet_name=sheet_name, skiprows=skip_rows)
        else:
            return pd.read_excel(path, skiprows=skip_rows)

    def clean_data(self, data):
        data.iloc[:, 0] = data.iloc[:, 0].str.replace('∟', '', regex=True)
        data.replace('\xa0', '', regex=True, inplace=True)
        data.replace('n.a.', np.nan, inplace=True)
        data = data.dropna(how='all', axis=1)
        data = data.dropna(thresh=10)
        return data

    def update_column_names(self, data, base_year):
        number_of_years = data.shape[1] - 1
        years = [str(base_year - i) for i in range(number_of_years)]
        data.columns = ['Cash Metric'] + years
        data['Cash Metric'] = data['Cash Metric'].str.strip()
        return data

    def pivot_data(self, data, pivot_column):
        data_pivot = data.pivot_table(values=data.columns[1:], columns=data[pivot_column])
        data_pivot.columns.name = "Columns"
        data_pivot.index.name = "Year"
        return data_pivot

    def merge_data_frames(self, *data_frames, on='Year', how='left'):
        merged_data = pd.DataFrame()
        for df in data_frames:
            if merged_data.empty:
                merged_data = df
            else:
                merged_data = pd.merge(merged_data, df, on=on, how=how)
        return merged_data

    def perform_regression_analysis(self, data, predictors, response):
        X = data[predictors]
        y = data[response]
        model = sm.OLS(y, X).fit()
        return model

# Usage
data_processor = DataProcessor("/Users/myself/Desktop/Walmart USA Searching for Growth/")
cashflow_data = data_processor.load_excel_data("walmartCashFlow.xlsx", skiprows=15)
cleaned_cashflow = data_processor.clean_data(cashflow_data)
updated_cashflow = data_processor.update_column_names(cleaned_cashflow, 2022)
cashflow_pivot = data_processor.pivot_data(updated_cashflow, 'Cash Metric')

store_data = data_processor.load_excel_data("walmartCashFlow.xlsx", "Yearly Store Count by Type", skiprows=2)
store_pivot = data_processor.pivot_data(store_data, 'Year')

merged_data = data_processor.merge_data_frames(cashflow_pivot, store_pivot)
model_results = data_processor.perform_regression_analysis(merged_data, ['Total SqFt Thousands'], 'Capital Expenditures')
