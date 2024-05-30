#!/usr/bin/env python
# coding: utf-8

import numpy as np

def estimate_visitor_distribution(total_visitors, prob_matrix, distances, densities):
    """
    Estimate visitor distribution across distances and densities.

    Parameters:
    total_visitors (int): Total number of visitors.
    prob_matrix (np.array): Probability matrix of visitors by distance and density.
    distances (np.array): Array of distances.
    densities (np.array): Array of densities.

    Returns:
    tuple: Estimated visitors by distance and visitors per density.
    """
    density_norm_factor = prob_matrix.sum(axis=0)
    density_norm = densities * density_norm_factor
    density_proportions = density_norm / np.sum(density_norm)
    visitors_per_density = density_proportions * total_visitors

    estimated_visitors_by_distance = np.zeros(len(distances))
    for i in range(len(densities)):
        estimated_visitors_by_distance += prob_matrix[:, i] * visitors_per_density[i]

    estimated_visitors_by_distance *= (total_visitors / np.sum(estimated_visitors_by_distance))

    return estimated_visitors_by_distance, visitors_per_density

def estimate_visitor_distribution_by_store_type(total_visitors, prob_matrix, distances, densities, store_data, num_simulations=1000):
    """
    Estimate visitor distribution by store type using Monte Carlo simulations.

    Parameters:
    total_visitors (int): Total number of visitors.
    prob_matrix (np.array): Probability matrix of visitors by distance and density.
    distances (np.array): Array of distances.
    densities (np.array): Array of densities.
    store_data (dict): Dictionary with store-specific data.
    num_simulations (int): Number of simulations to run.

    Returns:
    tuple: Estimated visitors by distance, visitors per density, visitors by store type, and simulation results.
    """
    visitor_results = []
    for _ in range(num_simulations):
        adjustment_factor = 1 + (np.random.rand(*prob_matrix.shape) - 0.5) * 0.2
        adjusted_matrix = prob_matrix * adjustment_factor
        density_norm_factor = adjusted_matrix.sum(axis=0)
        density_norm = densities * density_norm_factor
        density_proportions = density_norm / np.sum(density_norm)
        
        visitors_per_density = density_proportions * total_visitors

        total_visitors_by_type = {}     
        for store_type, data in store_data.items():
            footprint_ratio = data['avg_sqft'] / np.mean([d['avg_sqft'] for d in store_data.values()])
            location_factor = data['locations'] / sum(d['locations'] for d in store_data.values())
            eligible_visitors_per_density = visitors_per_density * data['density_eligibility']
            eligible_total_visitors = eligible_visitors_per_density.sum()
            raw_share = eligible_total_visitors * footprint_ratio * location_factor
            total_visitors_by_type[store_type] = raw_share
        
        sum_raw_shares = sum(total_visitors_by_type.values())
        normalization_factor = total_visitors / sum_raw_shares
        for store_type in total_visitors_by_type:
            total_visitors_by_type[store_type] *= normalization_factor
        
        visitor_results.append(list(total_visitors_by_type.values()))

        estimated_visitors_by_distance = np.zeros(len(distances))
        for store_type, visitors in total_visitors_by_type.items():
            for i, is_eligible in enumerate(store_data[store_type]['density_eligibility']):
                if is_eligible:
                    estimated_visitors_by_distance += prob_matrix[:, i] * (eligible_visitors_per_density[i] * normalization_factor)

    return estimated_visitors_by_distance, visitors_per_density, total_visitors_by_type, visitor_results

def monte_carlo_spending_adjustment(visitors_data, base_spending_per_visitor, scaling_factor_ranges, n_simulations, total_revenue):
    """
    Perform Monte Carlo simulation to adjust spending per visitor.

    Parameters:
    visitors_data (dict): Dictionary with visitor data.
    base_spending_per_visitor (dict): Base spending per visitor for each store type.
    scaling_factor_ranges (dict): Ranges for scaling factors.
    n_simulations (int): Number of simulations to run.
    total_revenue (float): Total revenue.

    Returns:
    list: Results of the simulation.
    """
    results = []

    for _ in range(n_simulations):
        temp_spending = {store_type: base_spending_per_visitor[store_type] * np.random.uniform(*scaling_factor_ranges[store_type])
                         for store_type in base_spending_per_visitor}
        
        normalized_spending, normalized_total, percent_spending = normalize_spending(visitors_data, temp_spending, total_revenue)
        results.append((normalized_spending, normalized_total, percent_spending))
    
    return results

def normalize_spending(visitors_data, base_spending_per_visitor, total_revenue):
    """
    Normalize spending per visitor to match total revenue.

    Parameters:
    visitors_data (dict): Dictionary with visitor data.
    base_spending_per_visitor (dict): Base spending per visitor for each store type.
    total_revenue (float): Total revenue.

    Returns:
    tuple: Normalized spending per visitor, normalized total spending, and percent spending per store.
    """
    initial_total_spending = sum(base_spending_per_visitor[store_type] * data['total_visitors']
                                 for store_type, data in visitors_data.items())
    
    normalization_factor = total_revenue / initial_total_spending
    
    normalized_spending_per_visitor = {store_type: base_spending_per_visitor[store_type] * normalization_factor
                                       for store_type in base_spending_per_visitor}
    
    normalized_spending_total = sum(normalized_spending_per_visitor[store_type] * data['total_visitors']
                                    for store_type, data in visitors_data.items())

    normalized_spending_per_store = {store_type: normalized_spending_per_visitor[store_type] * data['total_visitors']
                                       for store_type, data in visitors_data.items()}
    
    percent_spending_per_store = {store_type: normalized_spending_per_store[store_type] / normalized_spending_total
                                       for store_type in normalized_spending_per_store}
    
    return normalized_spending_per_visitor, normalized_spending_total, percent_spending_per_store

def calculate_averages(simulation_results, n_simulations):
    """
    Calculate averages from simulation results.

    Parameters:
    simulation_results (list): List of simulation results.
    n_simulations (int): Number of simulations run.

    Returns:
    tuple: Average normalized spending per store, average percent spending per store, and average normalized spending per visitor.
    """
    total_normalized_spending_per_store = {store_type: 0 for store_type in simulation_results[0][0]}
    total_percent_spending_per_store = {store_type: 0 for store_type in simulation_results[0][0]}
    total_normalized_spending_per_visitor = {store_type: 0 for store_type in simulation_results[0][0]}
    
    for normalized_spending, normalized_total, percent_spending in simulation_results:
        for store_type in total_normalized_spending_per_store:
            total_normalized_spending_per_store[store_type] += normalized_spending[store_type] * visitors_data[store_type]['total_visitors']
            total_percent_spending_per_store[store_type] += percent_spending[store_type]
            total_normalized_spending_per_visitor[store_type] += normalized_spending[store_type]
            
    avg_normalized_spending_per_store = {store_type: total / n_simulations for store_type, total in total_normalized_spending_per_store.items()}
    avg_percent_spending_per_store = {store_type: total / n_simulations for store_type, total in total_percent_spending_per_store.items()}
    avg_normalized_spending_per_visitor = {store_type: total / n_simulations for store_type, total in total_normalized_spending_per_visitor.items()}
    
    return avg_normalized_spending_per_store, avg_percent_spending_per_store, avg_normalized_spending_per_visitor

if __name__ == "__main__":
    # Define the probability matrix, distances, and densities
    prob_matrix = np.array([
        [0.999, 0.989, 0.966, 0.906, 0.717, 0.496, 0.236],
        [0.999, 0.979, 0.941, 0.849, 0.610, 0.387, 0.172],
        [0.997, 0.962, 0.899, 0.767, 0.490, 0.288, 0.123],
        [0.995, 0.933, 0.834, 0.659, 0.372, 0.206, 0.086],
        [0.989, 0.883, 0.739, 0.531, 0.268, 0.142, 0.060],
        [0.978, 0.803, 0.615, 0.398, 0.184, 0.096, 0.041]
    ])
    distances = np.array([0, 1, 2, 3, 4, 5])  # in miles
    densities = np.array([1, 5, 10, 20, 50, 100, 250])  # thousands of people within a 5-mile radius

    total_visitors = 1000000  # total number of visitors
    estimated_visitors, visitors_from_each_density = estimate_visitor_distribution(total_visitors, prob_matrix, distances, densities)

    print("Estimated Number of Visitors from Each Distance:")
    print(estimated_visitors)
    print("\nTotal Estimated Visitors by Distance:", np.sum(estimated_visitors))

    # Print total visitors from each density
    print("\nTotal Visitors from Each Density:")
    for density, visitors in zip(densities, visitors_from_each_density):
        print(f"Density {density}k: {visitors:.0f} visitors")

    print("\nTotal Estimated Visitors by Density:", np.sum(visitors_from_each_density))

    # Scaling expenditure per consumer by square footage of store type
    total_weekly_visits = 167 * 1e6
    total_annual_visits = 167 * 1e6 * 52
    total_net_sales = 393247 * 1e6 
    avg_spend = (393247 * 1e6) / (8684 * 1e6)

    store_data = {
        'Supercenters': {'locations': 3573, 'avg_sqft': 178000, 'density_eligibility': np.array([1, .8, .8, .8, 0.3, 0, 0])},
        'Discount Stores': {'locations': 370, 'avg_sqft': 105000, 'density_eligibility': np.array([0, .2, .2, .2, 0, 0, 0])},
        'Neighborhood Markets': {'locations': 799, 'avg_sqft': 42000, 'density_eligibility': np.array([0, 0, 0, 0, 0.7, 1, 1])}
    }

    estimated_visitors, visitors_from_each_density, visitors_by_store_type, visitor_results = estimate_visitor_distribution_by_store_type(
        total_visitors=total_annual_visits, prob_matrix=prob_matrix, distances=distances, densities=densities, store_data=store_data)

    print("Estimated Number of Visitors from Each Distance:")
    print(estimated_visitors)
    print("\nTotal Estimated Visitors by Distance:", np.sum(estimated_visitors))

    print("\nTotal Visitors from Each Density:")
    for density, visitors in zip(densities, visitors_from_each_density):
        print(f"Density {density}k: {visitors:.0f} visitors")

    print("\nVisitors by Store Type:")
    for store_type, visitors in visitors_by_store_type.items():
        print(f"{store_type}: {visitors:.0f} visitors")

    visitors_per_store_by_type = {}
    for store_type, visitors in visitors_by_store_type.items():
        visitors_per_store = visitors / store_data[store_type]['locations']
        print(f"{store_type}: {visitors:.0f} total visitors, {visitors_per_store:.0f} visitors per store")
        visitors_per_store_by_type[store_type] = visitors_per_store

    print("\nTotal Estimated Visitors by Store Type:", sum(visitors_by_store_type.values()))
    print("\nTotal Estimated Visitors by Density:", np.sum(visitors_from_each_density)) 

    visitor_results = np.array(visitor_results)
    print("Average Estimated Visitors:", np.mean(visitor_results[:,0]), np.mean(visitor_results[:,1]), np.mean(visitor_results[:,2]))
    print("Standard Deviation of Visitors:", np.std(visitor_results[:,0]), np.std(visitor_results[:,1]), np.std(visitor_results[:,2]))

    visitors_data = {
        'Supercenters': {'total_visitors': 7591667114, 'visitors_per_store': 2124732},
        'Discount Stores': {'total_visitors': 84852921, 'visitors_per_store': 229332},
        'Neighborhood Markets': {'total_visitors': 1007479965, 'visitors_per_store': 1260926}
    }
    total_revenue = total_net_sales

    base_spending_per_visitor = {
        'Supercenters': avg_spend,
        'Discount Stores': avg_spend,
        'Neighborhood Markets': avg_spend
    }

    scaling_factor_ranges = {
        'Supercenters': (1.2, 1.4),
        'Discount Stores': (.5, 1.5),
        'Neighborhood Markets': (1, 1.7)
    }

    n_simulations = 10000

    simulation_results = monte_carlo_spending_adjustment(visitors_data, base_spending_per_visitor, scaling_factor_ranges, n_simulations, total_revenue)
    avg_normalized_spending_per_store, avg_percent_spending_per_store, avg_normalized_spending_per_visitor = calculate_averages(simulation_results, n_simulations)

    print("Average Normalized Spending per Store:")
    for store_type, avg_spending in avg_normalized_spending_per_store.items():
        print(f"{store_type}: ${avg_spending:.2f}")

    print("\nAverage Percent Spending by Store Type:")
    for store_type, avg_percent in avg_percent_spending_per_store.items():
        print(f"{store_type}: {avg_percent * 100:.2f}%")

    print("\nAverage Visitor Spending by Store Type:")
    for store_type, avg_spending in avg_normalized_spending_per_visitor.items():
        print(f"{store_type}: ${avg_spending:.2f}")

    average_results = np.mean([result[1] for result in simulation_results])
    print("Average Normalized Spending Total across Simulations:", average_results)
