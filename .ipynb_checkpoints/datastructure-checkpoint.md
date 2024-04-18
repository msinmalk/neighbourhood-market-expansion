
### `demographic_data`

This dictionary contains information related to demographics that affect the market potential for new or existing Neighborhood Markets.

```python
demographic_data = {
    "population_density": 1200,  # People per square mile
    "demographic_factors": {
        "age_18_65": 0.6,  # Percentage of population between 18 to 65 years
        "income_per_capita": 35000  # Average income per capita
    },
    "areas": [  # List of potential expansion areas with their demographics
        {
            "name": "Urban Area 1",
            "population_density": 1500,
            "income_level": "high"
        },
        {
            "name": "Urban Area 3",
            "population_density": 1200,
            "income_level": "medium"
        }
    ]
}
```

### `market_conditions`

This dictionary captures current market conditions that influence the performance and strategic decisions regarding Neighborhood Markets.

```python
market_conditions = {
    "e_commerce_growth": 0.05,  # Projected annual growth rate of e-commerce segment
    "urban_expansion_factor": 0.02,  # Factor to account for urban expansion impact on revenue
    "new_store_openings": 3,  # Number of new stores planned for opening
    "cannibalization_rate": 0.01,  # Estimated impact of new stores on existing stores' revenue
    "market_saturation": 0.4,  # Current market saturation level
    "distribution_efficiency": 0.75,  # Efficiency of distribution network
    "distribution_center_proximity": 0.8,  # Proximity of stores to distribution centers, scale 0 to 1
    "proximity_factor": 0.2,  # Factor to adjust operational costs based on distribution center proximity
    "current_market_share": 0.05,  # Current market share
    "target_market_share": 0.15,  # Target market share
    "community_opposition_strength": 0.3  # Strength of community opposition to new store openings
}
```

### `financials`

This dictionary includes key financial metrics used to assess the financial viability and performance of Neighborhood Markets.

```python
financials = {
    "spending_per_consumer": 600,  # Average spending per consumer
    "variable_costs": 200000,  # Total variable costs
    "fixed_costs": 100000,  # Total fixed costs
    "gross_margin": 0.3,  # Gross margin percentage
    "existing_store_sales": 1000000,  # Sales from existing stores
    "base_operational_cost": 600000  # Base operational cost before adjustments
}
```

### `investment_factors`

This dictionary details the investment strategies and factors, such as labor investment, influencing spending and operational decisions.

```python
investment_factors = {
    "labor_investment": {
        "factor": 0.1  # Factor representing the increase in spending due to labor investment
    }
}
```

### `historical_data`

Contains historical financial data necessary for calculating Compound Annual Growth Rates (CAGRs) for key financial metrics.

```python
historical_data = {
    "initial_revenue": 500000,  # Revenue at the start of the period
    "final_revenue": 1000000,  # Revenue at the end of the period
    "initial_cogs": 300000,  # COGS at the start of the period
    "final_cogs": 600000,  # COGS at the end of the period
    "initial_ecommerce": 50000,  # E-commerce sales at the start of the period
    "final_ecommerce": 150000,  # E-commerce sales at the end of the period
    "periods": 5  # Number of years over which the CAGR is calculated
}
```