#!/usr/bin/env python
# coding: utf-8

# In[1]:


def clean_cashflow_data(cashflow_data):
    # Replace a special character in the first column
    cashflow_data.iloc[:, 0] = cashflow_data.iloc[:, 0].str.replace('∟', '', regex=True)

    # Replace non-breaking space and 'n.a.' with NaN
    cashflow_data.replace('\xa0', '', regex=True, inplace=True)
    cashflow_data.replace('n.a.', np.nan, inplace=True)

    # Drop columns where all values are NaN
    cashflow_data = cashflow_data.dropna(how='all', axis=1)

    # Drop rows with less than 10 non-NaN values
    cashflow_data = cashflow_data.dropna(thresh=10)

    return cashflow_data


def update_column_names(cashflow_data, base_year):
    # Calculate the number of years by subtracting one to exclude the 'Cash Metric' column
    number_of_years = cashflow_data.shape[1] - 1
    
    # Create a list of years in reverse order starting from the base year
    years = [str(base_year - i) for i in range(number_of_years)]
    
    # Update DataFrame column names
    cashflow_data.columns = ['Cash Metric'] + years
    
    # Strip whitespace from the 'Cash Metric' column
    cashflow_data['Cash Metric'] = cashflow_data['Cash Metric'].str.strip()
    
    return cashflow_data






# In[2]:


import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Import data into pandas DataFrame
# Assuming financial_data and store_counts are already loaded into pandas DataFrames
walmartPath="/Users/myself/Desktop/Walmart USA Serching for Growth/"



cashflow_data=pd.read_excel(walmartPath+"walmartCashFlow.xlsx",skiprows=15)

store_count=pd.read_excel(walmartPath+"walmartCashFlow.xlsx", "Yearly Store Count by Type",skiprows=2)
store_count.columns=store_count.columns.astype(str)

# cashflow_data,store_count
distribution_count=pd.read_excel(walmartPath+"walmartCashFlow.xlsx", "Yearly DC", skiprows=1)

cashflow_data.iloc[:,0]=cashflow_data.iloc[:,0].str.replace('∟', '', regex=True)

cashflow_data.replace('\xa0', '', regex=True, inplace=True)
cashflow_data.replace('n.a.', np.nan, inplace=True)

cashflow_data=cashflow_data.dropna(how='all',axis=1)
cashflow_data=cashflow_data.dropna(thresh=10)
cashflow_data


# Correctly identify and assign unique years to each financial data column
# Assuming the first column after 'Financial Metric' is the most recent year and decrement for each column after
number_of_years = cashflow_data.shape[1] - 1  # Total columns minus the 'Financial Metric' column
base_year = 2022
years = [str(base_year - i) for i in range(number_of_years)]

# # Map the new year labels to the columns
cashflow_data.columns = ['Cash Metric'] + years
cashflow_data['Cash Metric']=cashflow_data['Cash Metric'].str.strip()



statement_data=pd.read_excel(walmartPath+"walmartHistoricalFinancials.xlsx",skiprows=15)

statement_data = clean_cashflow_data(statement_data)
statement_data = update_column_names(statement_data,2021)


# In[3]:


def pivot_year_columns_to_rows(df,column_names):
    df_pivot = df.pivot_table(values=df.columns[1:],columns=df[column_names])
    if column_names == "Year":
        df_pivot.columns.name="Columns"
        df_pivot.index.name="Year"
    else:
        df_pivot.index.name="Year"
    return df_pivot

cashflow_pivot=pivot_year_columns_to_rows(cashflow_data,'Cash Metric')  #cashflow_data.pivot_table(values=cashflow_data.columns,columns=cashflow_data['Cash Metric'])
cashflow_pivot=cashflow_pivot/1000

statement_pivot=pivot_year_columns_to_rows(statement_data,'Cash Metric')
statement_pivot=statement_pivot/1000


store_pivot=pivot_year_columns_to_rows(store_count,'Year')
store_pivot.rename(columns={"Total":"Total Store"},inplace=True)
store_pivot = store_pivot.shift(periods=0)

avg_store_sqft={'Supercenters': 182000, 'Neighborhood markets': 38000, 'Discount stores': 106000, 'Total Store': 1}
store_sqft=store_pivot*avg_store_sqft/1000
store_sqft['Total Store'] = store_sqft['Supercenters'] + store_sqft['Neighborhood markets'] + store_sqft['Discount stores']



# Prepare distribution centre table for merge
distribution_count.rename(columns={"Total":"Total DC","Total.1":"Total SqFt"},inplace=True)
dist_pivot=distribution_count.pivot_table(index='Year')
dist_pivot.index=dist_pivot.index.astype(str)

dist_pivot = dist_pivot.shift(periods=0)
dist_pivot['Total SqFt Thousands'] = dist_pivot['Total SqFt']/1000



# In[4]:


merge_data_left = pd.merge(cashflow_pivot, store_pivot, on='Year', how='left')
#merge_data_left = pd.merge(merge_data_left, dist_pivot ,on='Year',how='left')
merge_data_left = pd.merge(merge_data_left, statement_pivot ,on='Year',how='left')



fin_merge_corr = merge_data_left.corr()
output_file_path = '/Users/myself/Desktop/Walmart USA Serching for Growth/Statement StoreDist Correlations.xlsx'
fin_merge_corr.to_excel(output_file_path, sheet_name='Correlations')


#sqft = merge_data_left[['Capital Expenditures','Total SqFt','Discount stores', 'Supercenters','Neighborhood markets','Total Store']]
#sqft.interpolate(inplace=True)

print(f'Financial Correlations saved to {output_file_path}')


# In[5]:


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Visualize the relationship between Capex and Rolling SqFt
# sns.regplot(x='Total SqFt', y='Capital Expenditures', data=merge_data_left)
# plt.title('Rolling SqFt vs. Capital Expenditure')
# plt.show()

# sns.regplot(x='Total Store', y='Capital Expenditures', data=merge_data_left)
# plt.title('Store Change vs. Capital Expenditure')
# plt.show()



# In[6]:


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from expertForecastFunctionsCopy1 import *

# Assuming 'data' is your DataFrame
# Convert all columns to numeric, setting errors='coerce' will convert non-convertible types to NaN



merge_left_numeric = merge_data_left.apply(pd.to_numeric, errors='coerce')
merge_left_numeric.index = pd.to_datetime(merge_left_numeric.index, format='%Y', errors='coerce')
merge_left_numeric.interpolate(method='linear',inplace=True)
merge_left_numeric['Neighborhood markets'] = merge_left_numeric['Neighborhood markets'].fillna(0)
merge_left_numeric.interpolate(method='linear', fill_value="extrapolate", limit_direction="both",inplace=True)
# Identifying indices with NaN values
nan_indices = merge_left_numeric[merge_left_numeric.isna().any(axis=1)].index
# print(nan_indices)

# Identifying columns with NaN values for each row
nan_details = merge_left_numeric.apply(lambda row: row.index[row.isna()].tolist(), axis=1)
# print(nan_details)



store_ratio=pd.DataFrame((merge_left_numeric['Discount stores']/merge_left_numeric['Supercenters']) * (merge_left_numeric['Discount stores']+merge_left_numeric['Supercenters']))
store_ratio.rename(columns={0:"Store ratio"},inplace=True)
merge_left_numeric=pd.merge(merge_left_numeric, store_ratio ,on='Year',how='left')

dist_pivot_reindex = pd.merge(pd.DataFrame(merge_data_left.index), dist_pivot ,on='Year',how='left')
dist_pivot_reindex.set_index('Year',inplace=True)
dist_pivot_reindex.ffill(inplace=True)
dist_pivot_reindex


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def analyze_correlations(variateDFT):
    # Creating a correlation matrix and displaying it as a heatmap
    corr_matrix = variateDFT.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix with Heatmap')
    plt.show()

    # Checking for multicollinearity using Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data["feature"] = variateDFT.columns
    vif_data["VIF"] = [variance_inflation_factor(variateDFT.values, i) for i in range(len(variateDFT.columns))]
    print(vif_data)


analyze_correlations(merge_left_numeric[['Store ratio','Neighborhood markets','Inventories', 'Capital Expenditures']])


# ### Beginning of PCA Work

# In[8]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(merge_left_numeric)

# Apply PCA
pca = PCA()
pca.fit(df_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

# Transform data
df_pca = pca.transform(df_scaled)
print("PCA Components:", df_pca)

loadings = pca.components_

# Create a DataFrame for better visualization of loadings
loadings_df = pd.DataFrame(loadings, columns=merge_left_numeric.columns, index=[f'PC{i+1}' for i in range(len(loadings))])
print(loadings_df)


import matplotlib.pyplot as plt

# Assuming 'explained_variance' is your array of explained variance ratios from PCA
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# ### Switching from PCA to PLS with Distribution SqFt as response variable

# In[102]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
if 'Total Sqft' in merge_left_numeric.columns:
    merge_left_numeric=merge_left_numeric.drop('Total SqFt', axis=1)
    
# Example Data
X = merge_left_numeric  # assuming 'DependentVariable' is your dependent variable
Y = dist_pivot_reindex['Total SqFt Thousands']

# Standardize the data (important for modeling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y.values.reshape(-1, 1)).flatten()

# PLS model
pls = PLSRegression(n_components=7)  # You can adjust the number of components
pls.fit(X_scaled, Y_scaled)

# Viewing the components
print("PLS Components (Weights for each feature in components):")
print(pls.x_weights_)

# Transform data
X_pls = pls.transform(X_scaled)


# ### GPT Attemped auto reduction of correlations from original data set

# In[76]:


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

# Assuming 'data' is your DataFrame containing all variables
# Calculate correlation matrix
data=X.copy()
correlation_matrix = data.corr()

# Identify pairs of highly correlated variables (absolute correlation > 0.9)
high_corr_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns 
                   if i != j and abs(correlation_matrix.loc[i, j]) > 0.9]

# # Print highly correlated pairs
# print("Highly correlated pairs:", high_corr_pairs)

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["Variable"] = data.columns
vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]

# # Print VIF values
# print("VIF values:\n", vif_data)

# Remove variables with high VIF (e.g., VIF > 5)
# variables_to_keep = vif_data[vif_data["VIF"] <= 5]["Variable"]
# data_reduced = data[variables_to_keep]

# Alternatively, apply PCA to reduce dimensionality
# pca = PCA(n_components=0.95)  # Keep 95% of the variance
# data_pca = pca.fit_transform(data_reduced)

# Apply PLS
# pls = PLSRegression(n_components=2)  # Adjust the number of components as needed
# pls.fit(data_pca, target_variable)  # Assuming 'target_variable' is your response variable

# Continue with your analysis using the fitted PLS model


# In[77]:


import numpy as np

# Identify pairs of variables with perfect correlation
perfect_corr_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns 
                      if i != j and abs(correlation_matrix.loc[i, j]) == 1.0]

# Print pairs of variables with perfect correlation
print("Perfectly correlated pairs:", perfect_corr_pairs)

# # Drop one variable from each pair of perfectly correlated variables
# for var1, var2 in perfect_corr_pairs:
#     if var1 in data.columns:
#         data.drop(var1, axis=1, inplace=True)


# List of variables to drop based on perfect correlation analysis
variables_to_drop = [
    'Purchase of Fixed Assets', 
    'Total Cash Dividends Paid', 
    'Issuance (Retirement) of Stock, Net', 
    'Depreciation, Supplemental', 
    'Depreciation/Depletion', 
    'Other Financing Cash Flow', 
    'Operating revenue (Turnover)', 
    "Total shareholders' funds and liabilities"
]

# Drop the identified variables from the dataset
data_reduced = data.drop(columns=variables_to_drop)
correlation_matrix = data_reduced.corr()
perfect_corr_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns 
                      if i != j and abs(correlation_matrix.loc[i, j]) == 1.0]


# Print pairs of variables with perfect correlation
print("Perfectly correlated pairs:", perfect_corr_pairs)

print("Variables after dropping perfect correlations:")
print(data_reduced.columns)


# In[78]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to calculate VIF
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

# Recalculate VIF
vif_data = calculate_vif(data_reduced)
print(vif_data)


# In[79]:


# Set a higher initial threshold for VIF to retain more variables initially
high_vif_threshold = 50
vif_data_high_vif = vif_data[vif_data['VIF'] > high_vif_threshold]

# Identify variables to drop based on the new threshold
variables_to_drop_iteratively = vif_data_high_vif['Variable'].tolist()
data_reduced_iteratively = data_reduced.drop(columns=variables_to_drop_iteratively)


# In[80]:


# Calculate variance of each variable
variances = data_reduced.var()

# Identify variables with near zero variance
near_zero_var = [col for col in data_reduced.columns if variances[col] < 1e-5]

# Drop variables with near zero variance
data.drop(near_zero_var, axis=1, inplace=True)

print(f"Variables with near zero variance removed: {near_zero_var}")


# In[82]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data_reduced.columns
    vif_data["VIF"] = [variance_inflation_factor(data_reduced.values, i) for i in range(len(data_reduced.columns))]
    return vif_data

# Function to iteratively remove variables with high VIF
def remove_high_vif(data, threshold=100.0):
    while True:
        vif_data = calculate_vif(data)
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold and len(data.columns) > 31 or max_vif==float('inf'):
            # Drop the variable with the highest VIF
            drop_variable = vif_data[vif_data["VIF"] == max_vif]["Variable"].values[0]
            data.drop(drop_variable, axis=1, inplace=True)
            print(f"Dropped {drop_variable} with VIF {max_vif}")
        else:
            break
    return data

# Remove variables with high VIF iteratively
data_double_reduced = remove_high_vif(data_reduced)

# Print remaining variables after VIF reduction
print("Remaining variables after VIF reduction:\n", data_double_reduced.columns)
calculate_vif(data_double_reduced)


# In[83]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
if 'Total Sqft' in merge_left_numeric.columns:
    merge_left_numeric=merge_left_numeric.drop('Total SqFt', axis=1)
    
# Example Data
X = data_double_reduced  # assuming 'DependentVariable' is your dependent variable
Y = dist_pivot_reindex['Total SqFt Thousands']

# Standardize the data (important for modeling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y.values.reshape(-1, 1)).flatten()

# PLS model
pls = PLSRegression(n_components=7)  # You can adjust the number of components
pls.fit(X_scaled, Y_scaled)

# Viewing the components
print("PLS Components (Weights for each feature in components):")
print(pls.x_weights_)

# Transform data
X_pls = pls.transform(X_scaled)


# ### End of Auto Drop PLS Attempt

# ### Beginning of analysing original Variable relevance to PLS model 

# In[103]:


import numpy as np

def calculate_vip(pls_model):
    t = pls_model.x_scores_  # Scores
    w = pls_model.x_weights_  # Weights
    q = pls_model.y_loadings_  # Y loadings
    
    p, h = w.shape
    vips = np.zeros((p,))
    
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 * s[j] for j in range(h)])
        vips[i] = np.sqrt(p * np.sum(weight) / total_s)
        
    return vips

# Assume 'pls' is your fitted PLS model
vip_scores = calculate_vip(pls)
print(vip_scores)


import numpy as np
import pandas as pd

# Create a DataFrame for better visualization and manipulation
variables = [f'Variable {i+1}' for i in range(len(vip_scores))]
vip_df = pd.DataFrame({'VIP Score': vip_scores}, index=variables)

# Sort the DataFrame to see the most important variables at the top
vip_df_sorted = vip_df.sort_values(by='VIP Score', ascending=False)

# Filter to find highly influential variables
key_variables = vip_df_sorted[vip_df_sorted['VIP Score'] > 1]

key_variables.to_excel('/Users/myself/Desktop/Walmart USA Serching for Growth/keyVarStoreDistCash.xlsx')
print(key_variables)  # This will display the variables with VIP scores greater than 1

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming vip_scores is a numpy array of your VIP scores
plt.figure(figsize=(12, 6))
sns.histplot(vip_scores, bins=30, kde=True, color='blue')
plt.title('Distribution of VIP Scores')
plt.xlabel('VIP Score')
plt.ylabel('Frequency')
plt.axvline(x=1, color='red', linestyle='--', label='VIP=1 Threshold')
plt.legend()
plt.show()

# Additionally, you might want to plot them in a sorted manner
sorted_scores = np.sort(vip_scores)[::-1]  # Sort descending
plt.figure(figsize=(12, 6))
plt.plot(sorted_scores, marker='o')
plt.title('VIP Scores Ordered by Value')
plt.xlabel('Variable Index')
plt.ylabel('VIP Score')
plt.axhline(y=1, color='red', linestyle='--', label='VIP=1 Threshold')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the VIP scores
plt.figure(figsize=(10, 8))
sns.barplot(x='VIP Score', y=vip_df_sorted.index, data=vip_df_sorted, palette='viridis')
plt.title('VIP Scores for Variables in PLS Model')
plt.xlabel('VIP Score')
plt.ylabel('Variables')
plt.show()

len(key_variables)


# In[ ]:





# ### Beginning of Domain Knowledge Variable reduction for PLS Simplicity

# In[105]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume 'data' is your DataFrame and 'vip_scores' contains the VIP scores as a numpy array
high_vip_indices = np.where(vip_scores > 1)[0]
high_vip_data = X.iloc[:, high_vip_indices]

# Calculate correlation matrix
corr_matrix = high_vip_data.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of High VIP Score Variables')
plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'high_vip_data' is already defined and contains data from high VIP score variables
corr_matrix = high_vip_data.corr()

# Set a threshold for high correlations (e.g., above 0.7 or 0.8)
threshold = 0.8

# Filter the correlation matrix
high_corr = corr_matrix[abs(corr_matrix) > threshold]

selfcorrelations=high_corr[high_corr == 1].stack()

# Remove self-correlations and duplicates
high_corr = high_corr[high_corr != 1].stack().drop_duplicates()

# Create a DataFrame from the filtered correlations for easier handling
high_corr_df = pd.DataFrame(high_corr, columns=['Correlation'])
high_corr_df = high_corr_df.reset_index()
high_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

# Sort by absolute correlation value for better readability
high_corr_df['Abs Correlation'] = high_corr_df['Correlation'].abs()
high_corr_df = high_corr_df.sort_values(by='Abs Correlation', ascending=False)

# Display the sorted DataFrame
print(high_corr_df[['Variable 1', 'Variable 2', 'Correlation']])

# Optional: Plot a subset of the matrix with high correlations
# Select variables to display based on high correlation pairs
variables_to_plot = list(set(high_corr_df['Variable 1']).union(set(high_corr_df['Variable 2'])))
reduced_corr_matrix = high_vip_data[variables_to_plot].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(reduced_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Reduced Correlation Matrix of High VIP Score Variables')
plt.show()

correlationInventoryList = high_corr_df[['Variable 1', 'Variable 2', 'Correlation']]
correlationInventoryList.to_excel('/Users/myself/Desktop/Walmart USA Serching for Growth/correlationInventoryList.xlsx')



# In[94]:


variables_to_drop = set()

for var1, var2 in zip(high_corr_df['Variable 1'], high_corr_df['Variable 2']):
    if var1 not in variables_to_drop and var2 not in variables_to_drop:
        # Add var2 to the drop list; you could choose var1 instead based on your criteria
        variables_to_drop.add(var)

print("Columns to drop:", variables_to_drop)
len(variables_to_drop)
#high_vip_data_reduced = high_vip_data.drop(columns=list(variables_to_drop))


# In[239]:


type(selfcorrelations)
selfcorrdf=selfcorrelations.unstack()

metriccount=[]
for metric in selfcorrdf.index:
    currcount=selfcorrdf.loc[metric] == 1.0
    metriccount.append(currcount)


# In[ ]:


# metric_count = []

# # Iterate over each metric in the DataFrame index
# for metric in selfcorrdf.index:
#     # Find the metrics that have a correlation of 1.0 with the current metric
#     correlated_metrics = selfcorrdf.columns[selfcorrdf.loc[metric] == 1.0].tolist()
    
#     # Append a tuple of the metric and its correlated metrics to the list
#     metric_count.append((metric, correlated_metrics))


# In[223]:


# for dex, val in enumerate(X.columns):
#     print(dex)


# ### Reducing PLS reliance to only high VIP (Variable Importance Power?) set

# In[106]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(high_vip_data)  # Use only high VIP score variables
Y = dist_pivot_reindex['Total SqFt Thousands']  # Your target variable

# Define PLS model
pls = PLSRegression(n_components=7)  # Adjust based on your previous analysis

# Setup cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(pls, X_scaled, Y, cv=kf, scoring='neg_mean_squared_error')

# Average MSE
average_mse = -np.mean(cv_scores)
print("Average MSE:", average_mse)


# In[234]:


27846226.980349492 - 158734981.76158053


# ### Trying to interpret which combination of variables proprotionally construct the end number of PLS components. 

# In[109]:


import pandas as pd
import matplotlib.pyplot as plt

# Extracting loadings
loadings = pls.x_weights_  # Change to your variable name if different

# Creating a DataFrame for better visualization
loadings_df = pd.DataFrame(loadings, columns=[f'Component {i+1}' for i in range(loadings.shape[1])], index=[f'Variable {i+1}' for i in range(loadings.shape[0])])

# Plotting loadings
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('PLS Component Loadings')
plt.xlabel('Components')
plt.ylabel('Variables')
plt.show()

import pandas as pd

# Assume 'pls' is your fitted PLS model
loadings = pls.x_weights_  # Extract loadings from PLS model

# Create a DataFrame of loadings
loadings_df = pd.DataFrame(loadings, 
                           columns=[f'Component {i+1}' for i in range(loadings.shape[1])],
                           index=[f'Variable {i+1}' for i in range(loadings.shape[0])])

# Order the DataFrame by the absolute values of loadings for the first component as an example
# You can repeat this for each component or adapt as necessary
component = 'Component 1'  # Choose which component to sort by
ordered_loadings_df = loadings_df.iloc[:, loadings_df.columns.get_loc(component)].abs().sort_values(ascending=False)

# Display the ordered DataFrame
print(ordered_loadings_df)


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the ordered loadings for the first component
sns.barplot(x=ordered_loadings_df.values, y=ordered_loadings_df.index)
plt.title(f'Loadings Ordered by {component}')
plt.xlabel('Loadings')
plt.ylabel('Variables')
plt.show()




# ### Considering how many componenets to generate for the PLS fit

# In[108]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Initialize PLS with 10 components
pls = PLSRegression(n_components=10)

# Define MSE scorer
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform cross-validation
cv_scores = cross_val_score(pls, X_scaled, Y_scaled, cv=kf, scoring=mse_scorer)

# Calculate average MSE
average_mse = np.mean(-cv_scores)
print("Average MSE:", average_mse)

import matplotlib.pyplot as plt

mse_results = []
component_range = range(1, 11)  # Assuming you want to test from 1 to 10 components

for n in component_range:
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, X_scaled, Y_scaled, cv=kf, scoring=mse_scorer)
    mse_results.append(-np.mean(scores))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(component_range, mse_results, marker='o')
plt.title('MSE vs. Number of PLS Components')
plt.xlabel('Number of Components')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

component_range, mse_results


# ### End of PCA / PLS work

# In[ ]:


# df=variateDFT
# # Assuming 'df' contains your data with columns 'Year', 'Net_Income', and 'Capex'
# X = sm.add_constant(df['Net Income'])  # Adds a constant term to the predictor
# y = df['Capital Expenditures']

# model = sm.OLS(y, X).fit()
# df['Capex_Residual'] = model.resid  # Calculate and store residuals

# # Review model summary to validate the linear relationship
# print(model.summary())

import pandas as pd
import statsmodels.api as sm

def analyze_capex_relationship(data, net_income_col, capex_col):
    # Add a constant term to the predictor for the linear regression model
    #X = sm.add_constant(data[net_income_col])
    X = data[net_income_col]
    y = data[capex_col]

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()
    
    # Calculate and store residuals in the DataFrame
    data['Capex Residual '+net_income_col] = model.resid

    # Print the summary of the model to review the linear relationship
    print(model.summary())

    return data


# analyze_capex_relationship(merge_left_select, 'Net Income', 'Capital Expenditures')
# analyze_capex_relationship(merge_left_select, 'Total Store', 'Capital Expenditures')
# analyze_capex_relationship(merge_left_select, 'Total SqFt Thousands', 'Capital Expenditures')


import pandas as pd
import statsmodels.api as sm

def perform_ols_regression(data, predictors, response):
    """
    Performs a multivariable OLS regression.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the dataset.
    - predictors (list of str): The names of the predictor columns.
    - response (str): The name of the response column.

    Returns:
    - model_summary (str): A summary of the regression model.
    """
    # Selecting predictors and adding a constant for the intercept
    X = data[predictors]
    #X = sm.add_constant(X)

    # Selecting the response variable
    y = data[response]

    # Creating and fitting the OLS model
    model = sm.OLS(y, X).fit()

    # Returning the summary of the model
    return model

model1= perform_ols_regression(merge_left_numeric,['Total SqFt Thousands'], 'Capital Expenditures')
model2= perform_ols_regression(merge_left_numeric,['Store ratio','Neighborhood markets','Inventories','Depreciation'], 'Total SqFt Thousands')


#merge_left_numeric['Store ratio'],merge_left_numeric['Total Store'],merge_left_numeric['Discount stores'],merge_left_numeric['Supercenters'],merge_left_numeric['Neighborhood markets'],merge_left_numeric['Total SqFt Thousands']

#print(merge_left_numeric.columns.tolist())


# In[ ]:


model1.summary()


# In[ ]:


model2.summary()


# In[ ]:


# def analyze_capex_store_relationship(financial_data, store_counts):
#     # Extracting CapEx data and the relevant years
#     capex_data = financial_data[financial_data['Cash Metric'].isin(['Purchase of Fixed Assets', 'Capital Expenditures'])]
#     print(f"Before transposition: {capex_data.shape}")  # Check the shape before transposition
#     print(capex_data)  # View the content
    
#     capex_data = capex_data.drop('Cash Metric', axis=1).transpose()  # Transpose to make the years as rows
#     print(f"After transposition: {capex_data.shape}")  # Check the shape after transposition
#     print(capex_data)  # View the content
    
#     # Ensure that we have exactly two rows for 'Purchase of Fixed Assets' and 'Capital Expenditures'
#     if capex_data.shape[0] == 2:
#         capex_data.columns = ['Purchase of Fixed Assets', 'Capital Expenditures']
#     else:
#         raise ValueError("The CapEx data does not have the expected number of rows after transposition.")
    
#     # ... rest of the original function ...
    
#     return capex_data  # Temporarily return this to check the output

# # Call the function with the data
# capex_data_checked = analyze_capex_store_relationship(cashflow_data, store_count)


# In[ ]:


# import pandas as pd


# def preprocess_store_counts(store_counts):
#     # Ensure all data is numeric, converting non-numeric to NaN
#     store_counts = store_counts.apply(pd.to_numeric, errors='coerce')

#     # Calculate net change in stores per year for each type of store
#     store_counts_diff = store_counts.diff(axis=1)  # Calculate year-over-year difference
#     store_counts_diff = store_counts_diff.iloc[:, 1:]  # Exclude the first column as it will be NaN after diff
    
#     return store_counts_diff

# # Re-run the preprocessing with corrected data types
# # store_counts_diff = preprocess_store_counts(store_count)

# # Optionally, you can check if there are many NaNs, which might indicate many non-numeric entries
# # print(store_counts_diff.isna().sum())

# def correlate_capex_changes(store_counts_diff, financial_data):
#     # Focus on Capital Expenditures for simplicity
#     capex_data = financial_data.loc[financial_data['Cash Metric'].isin(['Capital Expenditures'])]
#     capex_data = capex_data.drop('Cash Metric', axis=1).transpose()
#     capex_data.columns = ['Capital Expenditures']  # Set proper column name after transpose

#     # Ensure data types are correct for analysis
#     capex_data = capex_data.astype(float)  # Convert CapEx data to float for calculations
#     store_counts_diff = store_counts_diff.astype(float)  # Also ensure store count diffs are float

#     # Align the indices of both DataFrames to ensure they match for correlation analysis
#     # Both should have years as indices and be aligned accordingly
#     if not capex_data.index.equals(store_counts_diff.columns):
#         # Assuming both indices are years and formatted similarly
#         # This step might need adjustment based on actual index formats
#         capex_data = capex_data.reindex(store_counts_diff.columns)
#         print(store_counts_diff)
#         store_counts_diff=store_counts_diff.T
#         store_counts_diff=store_counts_diff.shift(periods=-1)
#         store_counts_diff=store_counts_diff.T
#         print(store_counts_diff)
#     # Compute correlation matrix
#     correlation_matrix = pd.concat([capex_data.iloc[::-1], store_counts_diff.loc[3]], axis=1).corr()
#     print(capex_data)
#     return correlation_matrix

# # Assuming 'cashflow_data' and 'store_count' are your DataFrames loaded correctly
# store_counts_diff = preprocess_store_counts(store_count)

# correlation_results = correlate_capex_changes(store_counts_diff, cashflow_data)

# # Output results
# print(correlation_results)
# store_count_total=store_count.loc[3]


# In[ ]:


# # def correlate_capex_changes(store_counts_diff, financial_data):
# #     # Focus on Capital Expenditures for simplicity
# capex_data = cashflow_data.loc[cashflow_data['Cash Metric'].isin(['Capital Expenditures'])]
# capex_data = capex_data.drop('Cash Metric', axis=1).transpose()
# capex_data.columns = ['Capital Expenditures']  # Set proper column name after transpose
# capex_data.index
# print(capex_data)
# dist_counts_diff = preprocess_store_counts(totaldist_count.T)
# dist_counts_diffT=dist_counts_diff.T
# dist_counts_diffT=dist_counts_diffT.shift(periods=-5)
# dist_counts_diffT.index=dist_counts_diffT.index.astype('string')
# print(dist_counts_diffT)


# dist_counts_diffT = dist_counts_diffT.loc[dist_counts_diffT.index.isin(capex_data.index)]
# #dist_counts_diffT
# correlation_matrix = pd.concat([capex_data, dist_counts_diffT], axis=1).corr()
# correlation_matrix


#     # # Ensure data types are correct for analysis
#     # capex_data = capex_data.astype(float)  # Convert CapEx data to float for calculations
#     # store_counts_diff = store_counts_diff.astype(float)  # Also ensure store count diffs are float

#     # # Align the indices of both DataFrames to ensure they match for correlation analysis
#     # # Both should have years as indices and be aligned accordingly
#     # if not capex_data.index.equals(store_counts_diff.columns):
#     #     # Assuming both indices are years and formatted similarly
#     #     # This step might need adjustment based on actual index formats
#     #     #capex_data = capex_data.reindex(store_counts_diff.columns)
#     #     store_counts_diff=store_counts_diff.loc[store_counts['Cash Metric'].isin(['Capital Expenditures'])]
#     #     store_counts_diff=store_counts_diff.reindex(capex_data.index)
#     #     store_counts_diff=store_counts_diff.T
#     #     #print(store_counts_diff)
#     # # Compute correlation matrix
#     # correlation_matrix = pd.concat([capex_data, store_counts_diff], axis=1).corr()
#     # return correlation_matrix

# # Assuming 'cashflow_data' and 'store_count' are your DataFrames loaded correctly

# # correlation_results = correlate_capex_changes(dist_counts_diff, cashflow_data)

# # Output results
# #print(correlation_results)


# In[ ]:


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# def expanded_correlation_analysis(financial_data, metrics_list):
#     # Filter for relevant metrics
#     relevant_data = financial_data[metrics_list]
#     # Calculate correlations
#     correlation_matrix = relevant_data.corr()
    
#     # Plot the correlation matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
#     plt.title('Correlation Matrix for Financial Metrics')
#     plt.show()

#     return correlation_matrix

# # Define the list of metrics to include in the analysis
# metrics_to_analyze = [
#     'Capital Expenditures',
#     'Net Income',
#     'Total Cash from Operating Activities',
#     'Depreciation',
#     'Changes in Working Capital',
#     'Total Cash from Financing Activities'
# ]

# if cashflow_data.index.name != 'Cash Metric':
#     cashflow_datacopy=cashflow_data.copy
#     cashflow_data.set_index('Cash Metric', inplace=True)  # Setting 'Cash Metric' as index

# cashflow_dataT=cashflow_data.T
# cashflow_dataT.columns
# # cashflow_dataT['Net Income']

# # Assuming 'cashflow_data' is your DataFrame containing all financial metrics
# correlation_results = expanded_correlation_analysis(cashflow_dataT, metrics_to_analyze)
# print(correlation_results)



# In[ ]:


# def analyze_capex_store_relationship(financial_data, store_counts):
#     # Analyze the relationship between CapEx and store counts
#     # ...

# def model_capex_for_new_stores(financial_data, store_counts):
#     # Model CapEx for new stores using regression analysis or curve fitting
#     # ...

# def monte_carlo_simulation(model):
#     # Perform Monte Carlo simulation using the CapEx model
#     # ...

# def integrate_with_expansion_model(expansion_model, capex_model):
#     # Update NeighborhoodMarketExpansionModel with new CapEx estimates
#     # ...

# # Now let's run these functions in sequence

# preprocessed_financial_data = preprocess_data(financial_data, store_counts)
# capex_store_relationship = analyze_capex_store_relationship(preprocessed_financial_data, store_counts)
# capex_model = model_capex_for_new_stores(preprocessed_financial_data, store_counts)
# simulation_results = monte_carlo_simulation(capex_model)
# integrate_with_expansion_model(neighborhood_market_expansion_model, capex_model)


# ## Distribution Center Correlation

# In[ ]:


# import pandas as pd

# # Assuming 'distribution_center_data' is loaded into a DataFrame with a 'Total' column already calculated
# def preprocess_distribution_totals(distribution_center_data):
#     # Calculate year-over-year changes for the total count of distribution centers
#     distribution_diff = distribution_center_data['Total'].diff().dropna()  # Calculate and drop the first NaN result
#     return distribution_diff

# def correlate_distribution_totals_capex(distribution_diff, capex_data):
#     # Ensure capex_data is aligned and formatted correctly, focusing on 'Capital Expenditures'
#     capex_data = capex_data.loc[capex_data['Cash Metric'] == 'Capital Expenditures'].drop('Cash Metric', axis=1)
#     capex_data = capex_data.transpose()
#     capex_data.columns = ['CapEx']
    
#     # Ensure the years align between both datasets
#     common_years = distribution_diff.index.intersection(capex_data.index)
#     distribution_aligned = distribution_diff.loc[common_years]
#     capex_aligned = capex_data.loc[common_years]
    
#     # Compute correlation
#     correlation_result = distribution_aligned.corr(capex_aligned['CapEx'])
#     return correlation_result

# # Load your distribution center data and cash flow data correctly before calling these functions
# distribution_diff = preprocess_distribution_totals(distribution_center_data)
# correlation_result = correlate_distribution_totals_capex(distribution_diff, cashflow_data)

# # Print or visualize the result
# print("Correlation between total distribution center changes and CapEx:", correlation_result)


# ## Some Visual Trash

# In[ ]:


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# from sklearn.linear_model import LinearRegression

# # Analyzing the relationship between CapEx and store counts
# def analyze_capex_store_relationship(financial_data, store_counts):
#     # Here you can calculate correlations and perform regression analysis
#     # This is a placeholder for the real implementation
#     correlation = financial_data.corrwith(store_counts)
#     print("Correlation between CapEx and store counts:", correlation)
    
#     # For visualization, you could plot CapEx vs store counts
#     plt.scatter(store_counts, financial_data['Capital Expenditures'])
#     plt.xlabel('Store Counts')
#     plt.ylabel('Capital Expenditures')
#     plt.show()

# # Modeling CapEx for new stores using regression analysis
# def model_capex_for_new_stores(financial_data, store_counts):
#     # Prepare the data
#     X = store_counts.values.reshape(-1, 1)  # Features
#     y = financial_data['Capital Expenditures'].values      # Target variable
    
#     # Fit the linear regression model
#     model = LinearRegression()
#     model.fit(X, y)
    
#     return model

# # Performing Monte Carlo simulation using the CapEx model
# def monte_carlo_simulation(model, n_simulations=1000):
#     # Placeholder for real implementation
#     # Simulate different scenarios to project future CapEx
#     simulations = []
#     for _ in range(n_simulations):
#         simulated_store_count = np.random.normal(loc=store_count_mean, scale=store_count_std)
#         projected_capex = model.predict([[simulated_store_count]])
#         simulations.append(projected_capex)
    
#     return np.mean(simulations), np.std(simulations)

# # Assuming financial_data and store_counts DataFrames are already preprocessed and loaded
# capex_model = model_capex_for_new_stores(cashflow_data, store_count)
# mean_projection, std_deviation = monte_carlo_simulation(capex_model)
# print(f"Projected CapEx Mean: {mean_projection}, Standard Deviation: {std_deviation}")

