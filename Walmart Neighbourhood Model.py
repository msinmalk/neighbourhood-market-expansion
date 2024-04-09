#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Import necessary libraries
import pandas as pd

# Historical financial data for Walmart
# This data is a simplified representation based on the approach outlined and would normally be loaded from the provided financial model file
historical_financials = {
    "Year": ["2017", "2018", "2019", "2020", "2021"],
    "Total Revenue": [ 500343, 514405, 523964, 559151, 572754 ],  # in million USD
    "Cost of Sales": [ 373396, 385301, 394605, 420315, 429000 ]  # in million USD
    "eCommerce": [25100.00, 39700.00, 64900.00, 73200.00]}

# Convert dictionary to DataFrame
financials_df = pd.DataFrame(historical_financials)

# Calculate CAGR for Total Revenue and Cost of Sales
def calculate_cagr(final_value, initial_value, periods):
    return (final_value / initial_value) ** (1 / periods) - 1

# Total Revenue CAGR
total_revenue_cagr = calculate_cagr(
    final_value=financials_df["Total Revenue"].iloc[-1], 
    initial_value=financials_df["Total Revenue"].iloc[0], 
    periods=len(financials_df) - 1
)

# Cost of Sales CAGR
cost_of_sales_cagr = calculate_cagr(
    final_value=financials_df["Cost of Sales"].iloc[-1], 
    initial_value=financials_df["Cost of Sales"].iloc[0], 
    periods=len(financials_df) - 1
)

total_revenue_cagr, cost_of_sales_cagr


# In[13]:


historical_financials


# In[15]:




# Assumptions for the financial projections
initial_investment_per_store = 4.5  # in millions
operational_cost_per_store_annual = histo*revenue_per_store_first_year  # in millions
revenue_per_store_first_year = 2  # in millions
annual_revenue_growth = 0.05  # 5% annual growth in revenue per store
new_stores_each_year = 5  # Moderate expansion scenario
store_cannibalization_rate = 0.15  # 10% revenue cannibalization from existing stores
#closure_rate_supercenters = 0.02  # 2% of supercenters closed each year due to redundancy
#existing_supercenters = 500  # number of existing supercenters
annual_foot_traffic_increase = 0.02  # 2% increase in customer visits
#revenue_cannibalized_per_supercenter = 1  # in millions
e_commerce_2021=0.11 # beginning from 11% eCommerce share of revenue 
e_commerce_growth = 0.02  # 8% annual growth for e-Commerce & Technology

df = pd.DataFrame(historical_financials)

# Calculate baseline CAGR for Total Revenue and Cost of Sales
revenue_cagr = total_revenue_cagr
cogs_cagr = cost_of_sales_cagr

# Project the financials for 2023 to 2027 based on the CAGR
projection_years = [2022, 2023, 2024, 2025, 2026]
projected_financials = {}
last_year_revenue = df['Total Revenue'].iloc[-1]
last_year_cogs = df['Cost of Sales'].iloc[-1]

for i, year in enumerate(projection_years):
    new_stores = new_stores_each_year * (i + 1)  # Cumulative count of new stores each year
    # Adjusting the investment in new stores
    investment_in_new_stores = new_stores * initial_investment_per_store  # in millions
    
    # Additional revenue from new stores
    additional_revenue = new_stores * revenue_per_store_first_year * (1 + annual_revenue_growth) ** i
    
    # Adjusting for cannibalization and closures
    revenue_cannibalized = additional_revenue * store_cannibalization_rate
    #(existing_supercenters * store_cannibalization_rate * revenue_cannibalized_per_supercenter * (1 - closure_rate_supercenters) ** i)
    net_new_revenue = additional_revenue - revenue_cannibalized
    
    # Foot traffic increase impact on revenue
    foot_traffic_revenue_increase = last_year_revenue * annual_foot_traffic_increase
    
    # Projected Total Revenue for the year after considering new stores and foot traffic increase
    projected_revenue = last_year_revenue * (1 + revenue_cagr) + net_new_revenue + foot_traffic_revenue_increase
    
    # Operational costs for new stores
    operational_costs = additional_revenue*.8
    
    # Projected Cost of Sales for the year after considering operational costs
    projected_cogs = last_year_cogs * (1 + cogs_cagr)
    
    # Update last year's figures for the next iteration
    last_year_revenue = projected_revenue
    last_year_cogs = projected_cogs
    
    # Store the projected figures
    projected_financials[year] = {
        'New Stores': new_stores,
        'Investment in New Stores': investment_in_new_stores,
        'Additional Revenue from New Stores': additional_revenue,
        'Revenue Cannibalized': revenue_cannibalized,
        'Net New Revenue': net_new_revenue,
        'Foot Traffic Revenue Increase': foot_traffic_revenue_increase,
        'Projected Total Revenue': projected_revenue,
        'Operational Costs for New Stores': operational_costs,
        'Projected COGS': projected_cogs
    }

# Create a DataFrame from the projected financials
df_projections = pd.DataFrame(projected_financials).T

df_projections


# In[4]:


historical_financials


# In[22]:




# Assumptions for the financial projections
initial_investment_per_store = 4.5  # in millions (https://journeymanco.com/project/walmart-neighborhood-markets-2/)
revenue_per_store_first_year = 2  # in millions based on per sq foot revenue 2021 * avg sq footage of a neighnourhood market store
margin=.8 #historical_avg
operational_cost_per_store_annual = (1-margin)*revenue_per_store_first_year  # in millions

annual_revenue_growth = total_revenue_cagr  # conservative estimate of store rev growth on par with company wide annual growth in revenue per store
new_stores_each_year = 5  # Moderate expansion scenario
store_cannibalization_rate = 0.15  # 15% revenue cannibalization from existing stores + eCommerce
#revenue_cannibalized_per_supercenter = 1  # in millions
e_commerce_share=0.11 # beginning from 11% eCommerce share of revenue 2021
e_commerce_share_growth = 0.02  # 8% annual growth for e-Commerce & Technology
annual_e_commerce_increase = 0.02  # 2% increase in customer visits
per_weekly_customer_revenue= #https://www.yaguara.co/walmart-statistics/


df = pd.DataFrame(historical_financials)

# Calculate baseline CAGR for Total Revenue and Cost of Sales
revenue_cagr = total_revenue_cagr
cogs_cagr = cost_of_sales_cagr

# Project the financials for 2023 to 2027 based on the CAGR
projection_years = [2022, 2023, 2024, 2025, 2026]
projected_financials = {}
last_year_revenue = df['Total Revenue'].iloc[-1]
last_year_cogs = df['Cost of Sales'].iloc[-1]

for i, year in enumerate(projection_years):
    new_stores = new_stores_each_year * (i + 1)  # Cumulative count of new stores each year
    # Adjusting the investment in new stores
    investment_in_new_stores = new_stores * initial_investment_per_store  # in millions
    
    # Additional revenue from new stores
    additional_revenue = new_stores * revenue_per_store_first_year * (1 + annual_revenue_growth) ** i
    
    # Foot traffic eCommerce increase impact on revenue
    last_year_e_commerce = e_commerce_share * last_year_revenue
    e_commerce_share = e_commerce_share + e_commerce_share_growth
    net_annual_e_commerce = e_commerce_share * last_year_revenue * annual_e_commerce_increase
    
    
    # Adjusting for cannibalization
    revenue_cannibalized = additional_revenue * store_cannibalization_rate
    net_new_revenue = additional_revenue - revenue_cannibalized + net_annual_e_commerce

    # Projected Total Revenue for the year after considering new stores and foot traffic increase
    projected_revenue = last_year_revenue * (1 + revenue_cagr) + net_new_revenue
    
    # Operational costs for new stores
    operational_costs = additional_revenue*.8
    
    # Projected Cost of Sales for the year after considering operational costs
    projected_cogs = last_year_cogs * (1 + cogs_cagr)
    
    # Update last year's figures for the next iteration
    last_year_revenue = projected_revenue
    last_year_cogs = projected_cogs
    
    # Store the projected figures
    projected_financials[year] = {
        'New Stores': new_stores,
        'Investment in New Stores': investment_in_new_stores,
        'Additional Revenue from New Stores': additional_revenue,
        'Revenue Cannibalized': revenue_cannibalized,
        'Net New Revenue': net_new_revenue,
        'eCommerce Foot Traffic Revenue Increase': net_annual_e_commerce,
        'Projected Total Revenue': projected_revenue,
        'Operational Costs for New Stores': operational_costs,
        'Projected COGS': projected_cogs
    }

# Create a DataFrame from the projected financials
df_projections = pd.DataFrame(projected_financials).T

df_projections


# In[23]:


# Define placeholder functions for calculating factors based on density, distance, and e-commerce data.
# These functions need to be defined based on your specific data and model assumptions.

def calculate_density_factor(density):
    """
    Placeholder function to calculate the impact of population density on revenue.
    This could be defined based on empirical data or literature.
    """
    # Example: Higher density increases revenue potential up to a certain point.
    if density < 10:
        return 1 + (density * 0.05)
    elif density < 50:
        return 1.5
    else:
        return 1.2

def calculate_distance_factor(distance):
    """
    Placeholder function to calculate the impact of store distance on revenue.
    Assuming closer distance increases the revenue potential.
    """
    # Example: Decrease factor as distance increases.
    if distance <= 5:
        return 1.0
    elif distance <= 10:
        return 0.9
    else:
        return 0.8

def calculate_ecommerce_factor(e_commerce_data, density, distance):
    """
    Placeholder function to adjust revenue based on e-commerce data, taking into account
    population density and store distance.
    """
    # Example adjustment: Higher e-commerce share and closer distance increase revenue potential.
    base_increase = e_commerce_data['base_increase']
    adjustment_for_density = 0.01 * density  # Hypothetical adjustment
    adjustment_for_distance = -0.02 * max(0, (distance - 5))  # Decrease for distance > 5 miles
    return base_increase + adjustment_for_density + adjustment_for_distance

# Function to adjust revenue based on population density, store distance, and e-commerce
def adjust_revenue_for_ecommerce_and_density(store_data, density, distance, e_commerce_data):
    density_factor = calculate_density_factor(density)
    distance_factor = calculate_distance_factor(distance)
    e_commerce_factor = calculate_ecommerce_factor(e_commerce_data, density, distance)
    
    adjusted_revenue = store_data['base_revenue'] * density_factor * distance_factor + e_commerce_factor
    return adjusted_revenue

# Example usage
store_data = {'base_revenue': 2}  # in millions
density = 30  # thousands of people within a 5-mile radius
distance = 8  # miles from the distribution center
e_commerce_data = {'base_increase': 0.1}  # Hypothetical base increase in revenue from e-commerce

# Calculate the adjusted revenue
adjusted_revenue = adjust_revenue_for_ecommerce_and_density(store_data, density, distance, e_commerce_data)
adjusted_revenue


# In[25]:





# In[ ]:




