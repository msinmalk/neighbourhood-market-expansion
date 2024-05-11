import pandas as pd
import numpy as np

def clean_cashflow_data(cashflow_data, threshhold):
    # Replace a special character in the first column

    cashflow_data.columns=cashflow_data.columns.astype(str)
    cashflow_data.iloc[:, 0] = cashflow_data.iloc[:, 0].str.replace('∟', '', regex=True)

    # Replace non-breaking space and 'n.a.' with NaN
    cashflow_data.replace('\xa0', '', regex=True, inplace=True)
    cashflow_data.replace('n.a.', np.nan, inplace=True)
    cashflow_data.iloc[:, 0] = cashflow_data.iloc[:, 0].str.strip()
    # Drop columns where all values are NaN
    cashflow_data = cashflow_data.dropna(how='all', axis=1)

    # Drop rows with less than 10 non-NaN values
    cashflow_data = cashflow_data.dropna(thresh=threshhold)

    return cashflow_data


def update_column_names(cashflow_data, base_year, index_name):
    # Calculate the number of years by subtracting one to exclude the 'Cash Metric' column
    number_of_years = cashflow_data.shape[1] - 1
    
    # Create a list of years in reverse order starting from the base year
    years = [str(base_year - i) for i in range(number_of_years)]
    
    # Update DataFrame column names
    cashflow_data.columns = [index_name] + years
    
    # Strip whitespace from the 'Cash Metric' column
    cashflow_data.index=cashflow_data[index_name]
    cashflow_data=cashflow_data.drop([index_name])
    
    return cashflow_data


def pivot_year_columns_to_rows(df,pivot_column):
    df_pivot = df.pivot_table(values=df.columns[1:],columns=df[pivot_column])
    if pivot_column == "Year":
        df_pivot.columns.name="Columns"
        df_pivot.index.name="Year"
    else:
        df_pivot.index.name="Year"
    return df_pivot


