import unittest
from NeighborhoodMarketExpansionModel import NeighborhoodMarketExpansionModel

class TestNeighborhoodMarketExpansionModel(unittest.TestCase):
    def setUp(self):
        # Setup that will be run before each test method
        self.financials = {'Revenue': 1000000, 'Capital Expenditures': 150000, 'Operating Expenses': 200000}
        self.store_data = {'NeighborhoodMarket': {'locations': 50, 'average_labor_cost': 30000, 'current_distribution_sqft': 50000}}
        self.prob_matrix = [[0.7, 0.3], [0.4, 0.6]]
        self.distances = [5, 10, 15]
        self.densities = [1000, 1500, 1200]
        self.avg_spending_per_visitor = {'NeighborhoodMarket': 45.50}
        self.financial_forecast = {'Revenue Forecast': 1200000, 'Capital Expenditures Forecast': 200000}
        self.regression_results = {'sqft_per_store': 1500, 'cost_per_sqft': 10}
        self.market_conditions = {'community_opposition': 0.5, 'cannibalization_rate': 0.01, 'ecommerce_growth_factor': 0.03}

        self.model = NeighborhoodMarketExpansionModel(
            financials=self.financials,
            store_data=self.store_data,
            prob_matrix=self.prob_matrix,
            distances=self.distances,
            densities=self.densities,
            avg_spending_per_visitor=self.avg_spending_per_visitor,
            financial_forecast=self.financial_forecast,
            regression_results=self.regression_results,
            market_conditions=self.market_conditions
        )

    def test_simulate_store_numbers(self):
        # Test to ensure that store number simulation stops correctly
        visitors_by_store_type = {'NeighborhoodMarket': 1000}
        optimal_visitors = 100
        max_stores = 20
        results = self.model.simulate_store_numbers(visitors_by_store_type, optimal_visitors, max_stores, 'NeighborhoodMarket')
        self.assertEqual(len(results), 11)  # Expecting 11 because it should stop adding stores when profitability drops below optimal

    def test_estimate_new_distribution_sqft(self):
        # Test to ensure correct estimation of new distribution square footage
        num_new_stores = 10
        current_total_stores = 50
        expected_sqft = 1500 * (60) - 50000
        actual_sqft = self.model.estimate_new_distribution_sqft(num_new_stores, current_total_stores)
        self.assertEqual(actual_sqft, expected_sqft)

    def test_adjust_operational_costs(self):
        # Test adjustment of operational costs based on market conditions
        base_cost = 100000
        expected_cost = base_cost * (1 - 0.2 * 0.5)
        actual_cost = self.model.adjust_operational_costs(base_cost)
        self.assertAlmostEqual(actual_cost, expected_cost)

if __name__ == '__main__':
    unittest.main()



# import classTestData
# unittest.main(module='classTestData', argv=[''], exit=False)
