class NeighborhoodMarketExpansionModel:
    def __init__(self, curr_finances, store_data, prob_matrix, distances, densities, visitor_estimates, avg_spending_per_visitor, financial_forecast, regression_results, market_conditions):
        self.curr_finances = curr_finances
        self.store_data = store_data
        self.prob_matrix = prob_matrix
        self.distances = distances
        self.densities = densities
        self.avg_spending_per_visitor = avg_spending_per_visitor
        self.financial_forecast = financial_forecast
        self.regression_results = regression_results
        self.market_conditions = market_conditions  # Correctly initializing the market_conditions
        self.visitor_estimates = visitor_estimates


    # def estimate_visitors(self, total_visitors):
    #     # Call the visitor distribution estimation function
    #     estimated_visitors, visitors_from_each_density, visitors_by_store_type, visitor_results = estimate_visitor_distribution_by_store_type(
    #         total_visitors=total_visitors, prob_matrix=self.prob_matrix, distances=self.distances, densities=self.densities, store_data=self.store_data)
    #     self.visitor_estimates = {
    #         'estimated_visitors': estimated_visitors,
    #         'visitors_from_each_density': visitors_from_each_density,
    #         'visitors_by_store_type': visitors_by_store_type,
    #         'visitor_results': visitor_results
    #     }
    #     return self.visitor_estimates
    
    def simulate_store_numbers(self, visitors_by_store_type, optimal_visitors, max_stores, store_type):
        # corrected to include 'self' and proper attribute access
        results = {}
        for num_stores in range(1, max_stores + 1):
            visitors_per_store = visitors_by_store_type[store_type] / (self.store_data[store_type]['locations'] + num_stores)
            if visitors_per_store < optimal_visitors:
                profitability = visitors_per_store / optimal_visitors * 100  # as percentage
            else:
                profitability = 100

            results[num_stores] = (visitors_per_store, profitability)
            if visitors_per_store < optimal_visitors:
                break  # Stop if stores are not reaching optimal visitor count

        return results
    
    def estimate_new_distribution_sqft(self, num_new_stores, current_total_stores):
        expected_total_sqft = self.regression_results['sqft_per_store'] * (current_total_stores + num_new_stores)
        current_sqft = self.store_data['current_distribution_sqft']
        net_new_sqft = expected_total_sqft - current_sqft
        return net_new_sqft

    def calculate_cannibalization_effect(self, existing_store_sales, new_store_openings):
        cannibalization_rate = self.market_conditions.get('cannibalization_rate', 0.01)
        adjusted_sales = existing_store_sales * (1 - cannibalization_rate * new_store_openings)
        return adjusted_sales

    def dynamic_ecommerce_growth(self, current_sales):
        ecommerce_growth_factor = self.market_conditions.get('ecommerce_growth_factor', 0.05)
        market_saturation = self.market_conditions.get('market_saturation', 0.1)
        distribution_efficiency = self.market_conditions.get('distribution_efficiency', 0.9)
        adjusted_growth_rate = ecommerce_growth_factor * (1 - market_saturation) * distribution_efficiency
        adjusted_sales = current_sales * (1 + adjusted_growth_rate)
        return adjusted_sales

    def adjust_operational_costs(self, base_cost):
        proximity_factor = self.market_conditions.get('proximity_factor', 0.2)
        distribution_center_proximity = self.market_conditions.get('distribution_center_proximity', 0.5)
        adjusted_cost = base_cost * (1 - proximity_factor * distribution_center_proximity)
        return adjusted_cost

    def labor_investment_needed(self, current_market_share, target_market_share, community_opposition_strength):
        labor_investment_increase = 0
        while current_market_share < target_market_share:
            current_market_share *= (1 + 0.01 * (1 - community_opposition_strength))
            labor_investment_increase += 0.01
        return labor_investment_increase

    def evaluate_labor_investment_impact(self, labor_costs, initial_profit_margin, labor_investment_increase):
        adjusted_labor_costs = labor_costs * (1 + labor_investment_increase)
        adjusted_profit_margin = initial_profit_margin * (1 - 0.05 * labor_investment_increase)
        return adjusted_labor_costs, adjusted_profit_margin

    def estimate_new_store_costs(self, num_new_stores, store_type, current_total_stores):
        net_new_sqft = self.estimate_new_distribution_sqft(num_new_stores, current_total_stores)
        capital_costs = net_new_sqft * self.regression_results['cost_per_sqft']
        average_labor_cost = self.store_data[store_type]['average_labor_cost']
        total_labor_cost = average_labor_cost * num_new_stores
        opposition_factor = self.market_conditions.get('community_opposition', 1.0)
        adjusted_capital_costs = capital_costs * opposition_factor
        adjusted_labor_costs = total_labor_cost * opposition_factor
        return adjusted_capital_costs, adjusted_labor_costs

    def simulate_store_impact(self, max_new_stores, store_type, current_total_stores):
        for num_stores in range(1, max_new_stores + 1):
            self.update_financial_forecasts(num_stores, store_type, current_total_stores)
            print(f"After adding {num_stores} {store_type} stores: Updated Financial Forecast")

    def update_financial_forecasts(self, num_new_stores, store_type, current_total_stores):
        capital_costs, labor_costs = self.estimate_new_store_costs(num_new_stores, store_type, current_total_stores)
        self.financial_forecast['Capital Expenditures'] += capital_costs
        self.financial_forecast['Operating Expenses'] += labor_costs



    def run_monte_carlo_simulations(self, num_simulations=1000):
        results = []
        for _ in range(num_simulations):
            store_type = np.random.choice(list(self.avg_spending_per_visitor.keys()))  # Randomly choose a store type
            avg_spending = self.avg_spending_per_visitor[store_type]
            current_visitors = np.random.normal(10000, 2000)  # Simulated current visitors for the store type
            current_sales = current_visitors * avg_spending  # Calculate sales based on average spending per visitor

            new_store_openings = np.random.randint(1, 10)  # Simulated new store openings
            adjusted_sales = self.calculate_cannibalization_effect(current_sales, new_store_openings)
            adjusted_sales = self.dynamic_ecommerce_growth(adjusted_sales)
            adjusted_costs = self.adjust_operational_costs(self.curr_finances['base_operational_cost'])
            net_profit = adjusted_sales - adjusted_costs
            results.append(net_profit)
        average_profit = np.mean(results)
        profit_std_dev = np.std(results)
        return average_profit, profit_std_dev


class ECommerceGrowthModel(NeighborhoodMarketExpansionModel):
    def __init__(self, base_sales, subscribers, subscription_fee, fulfillment_impact, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_sales = base_sales
        self.subscribers = subscribers
        self.subscription_fee = subscription_fee
        self.fulfillment_impact = fulfillment_impact

    def calculate_incremental_revenue_from_subscriptions(self):
        direct_revenue = self.subscribers * self.subscription_fee
        indirect_revenue = self.base_sales * 0.05  # Assumed increase in purchases
        return direct_revenue + indirect_revenue

    def calculate_revenue_from_fulfillment_expansion(self):
        additional_revenue = self.base_sales * self.fulfillment_impact
        return additional_revenue

    def simulate_e_commerce_growth(self):
        subscription_revenue = self.calculate_incremental_revenue_from_subscriptions()
        fulfillment_revenue = self.calculate_revenue_from_fulfillment_expansion()
        total_revenue = self.base_sales + subscription_revenue + fulfillment_revenue
        return total_revenue


def estimate_visitor_distribution_by_store_type(total_visitors, prob_matrix, distances, densities, store_data, num_simulations=1000):
    # Calculate the proportion of total visitors from each density based on their relative values
    # density_norm_factor = prob_matrix.sum(axis=0)
    # density_norm = densities * density_norm_factor
    # density_proportions = density_norm / np.sum(density_norm)

    # Calculate the total number of visitors attributed to each density
    visitor_results=[]
    for _ in range(num_simulations):
        # Randomly adjust probabilities by up to ±10% to simulate changes in demographics or economic conditions
        adjustment_factor = 1 + (np.random.rand(*prob_matrix.shape) - 0.5) * 0.2
        adjusted_matrix = prob_matrix * adjustment_factor
        density_norm_factor = adjusted_matrix.sum(axis=0)
        density_norm = densities * density_norm_factor
        density_proportions = density_norm / np.sum(density_norm)
        
        visitors_per_density = density_proportions * total_visitors

    # Calculate total visitors based on store type and density eligibility
        total_visitors_by_type = {}     
        for store_type, data in store_data.items():
            # Adjust the total visitors according to the store type specifics
            footprint_ratio = data['avg_sqft'] / np.mean([d['avg_sqft'] for d in store_data.values()])
            location_factor = data['locations'] / sum(d['locations'] for d in store_data.values())
    
            # Apply density eligibility
            eligible_visitors_per_density = visitors_per_density * data['density_eligibility']
            eligible_total_visitors = eligible_visitors_per_density.sum()  # Sum of eligible visitors per density
    
            # Calculate the share of visitors for this store type
            raw_share = eligible_total_visitors * footprint_ratio * location_factor
            total_visitors_by_type[store_type] = raw_share
       
    
        # Normalize the visitor distribution to match the total number of visitors
        sum_raw_shares = sum(total_visitors_by_type.values())
        normalization_factor = total_visitors / sum_raw_shares
        for store_type in total_visitors_by_type:
            total_visitors_by_type[store_type] *= normalization_factor
        
        visitor_results.append(list(total_visitors_by_type.values()))

        # Calculate and distribute visitors
        estimated_visitors_by_distance = np.zeros(len(distances))
        for store_type, visitors in total_visitors_by_type.items():
            #visitor_results.append({store_type:visitors})
            for i, is_eligible in enumerate(store_data[store_type]['density_eligibility']):
                if is_eligible:
                    estimated_visitors_by_distance += prob_matrix[:, i] * (eligible_visitors_per_density[i] * normalization_factor)

    return estimated_visitors_by_distance, visitors_per_density, total_visitors_by_type, visitor_results


def monte_carlo_spending_adjustment(visitors_data, base_spending_per_visitor, scaling_factor_ranges, n_simulations, total_revenue):
    results = []

    for _ in range(n_simulations):
        # Apply random scaling factors to the base spending per visitor
        temp_spending = {store_type: base_spending_per_visitor[store_type] * np.random.uniform(*scaling_factor_ranges[store_type])
                         for store_type in base_spending_per_visitor}
        
        # Normalize the spending using your existing function
        normalized_spending, normalized_total, percent_spending = normalize_spending(visitors_data, temp_spending, total_revenue)
        results.append((normalized_spending, normalized_total, percent_spending))
    
    return results


def normalize_spending(visitors_data, base_spending_per_visitor, total_revenue):
    # Calculate initial total spending
    initial_total_spending = sum(base_spending_per_visitor[store_type] * data['total_visitors']
                                 for store_type, data in visitors_data.items())
    
    # Normalization factor to match the total revenue
    normalization_factor = total_revenue / initial_total_spending
    
    # Adjusted spending per visitor
    normalized_spending_per_visitor = {store_type: base_spending_per_visitor[store_type] * normalization_factor
                                       for store_type in base_spending_per_visitor}
    
    normalized_spending_total = sum(normalized_spending_per_visitor[store_type] * data['total_visitors']
                                    for store_type, data in visitors_data.items())

    normalized_spending_per_store = {store_type: normalized_spending_per_visitor[store_type] * data['total_visitors']
                                       for store_type, data in visitors_data.items()}
    
    percent_spending_per_store = {store_type: normalized_spending_per_store[store_type] / normalized_spending_total
                                       for store_type in normalized_spending_per_store}
    
    return normalized_spending_per_visitor, normalized_spending_total, percent_spending_per_store

def calculate_averages(simulation_results):
    # Initialize dictionaries to accumulate totals for averaging later
    total_normalized_spending_per_store = {store_type: 0 for store_type in visitors_data}
    total_percent_spending_per_store = {store_type: 0 for store_type in visitors_data}
    total_normalized_spending_per_visitor = {store_type: 0 for store_type in visitors_data}
    
    # Accumulate totals from each simulation
    for normalized_spending, normalized_total, percent_spending in simulation_results:
        for store_type in visitors_data:
            total_normalized_spending_per_store[store_type] += normalized_spending[store_type] * visitors_data[store_type]['total_visitors']
            total_percent_spending_per_store[store_type] += percent_spending[store_type]
            total_normalized_spending_per_visitor[store_type] += normalized_spending[store_type]
            
    # Calculate averages by dividing by the number of simulations
    avg_normalized_spending_per_store = {store_type: total / n_simulations for store_type, total in total_normalized_spending_per_store.items()}
    avg_percent_spending_per_store = {store_type: total / n_simulations for store_type, total in total_percent_spending_per_store.items()}
    avg_normalized_spending_per_visitor = {store_type: total / n_simulations for store_type, total in total_normalized_spending_per_visitor.items()}
    
    return avg_normalized_spending_per_store, avg_percent_spending_per_store, avg_normalized_spending_per_visitor





# class FinancialForecastModel:
#     def __init__(self, data_processor, expansion_model):
#         self.data_processor = data_processor
#         self.expansion_model = expansion_model

#     def perform_forecasts(self, financial_data):
#         # Generate forecasts using different methods
#         yoy_forecast = self.data_processor.yoy_forecast(financial_data)
#         regression_forecast = self.data_processor.regression_forecast(financial_data)
#         ratio_based_forecast = self.data_processor.ratio_forecast(financial_data)

#         # Return all forecasts
#         return {
#             'yoy': yoy_forecast,
#             'regression': regression_forecast,
#             'ratio_based': ratio_based_forecast
#         }

#     def run_sensitivity_analysis(self, forecasts):
#         results = {}
#         for method, forecast_data in forecasts.items():
#             # Configure the expansion model with forecast data
#             self.expansion_model.configure(forecast_data)

#             # Run the model and store results
#             results[method] = self.expansion_model.simulate()

#         return results

# # Usage
# data_processor = DataProcessor('/path/to/data')
# expansion_model = NeighborhoodMarketExpansionModel(...)
# forecast_model = FinancialForecastModel(data_processor, expansion_model)

# financial_data = pd.read_excel('/path/to/financial_data.xlsx')
# forecasts = forecast_model.perform_forecasts(financial_data)
# sensitivity_results = forecast_model.run_sensitivity_analysis(forecasts)
