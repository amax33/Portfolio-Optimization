# Portfolio Optimization Project
This project involves optimizing a portfolio of assets to maximize returns while minimizing risks. The code uses several optimization methods to achieve the best asset allocation under specified constraints.

## File Structure
├── calculator.py     # Core optimization logic using various algorithms

├── chart.py          # Visualization of risk/return trade-offs for different lambda values

└── Stock_Generator.py # Functions for generating and handling stock data and covariance matrices

## calculator.py
### Description: 
Implements the portfolio optimization problem using various algorithms and constraints.

### Classes:

#### Stock: 
   Defines attributes for stocks, including expected return, variance, and cost.
#### Market: 
   Manages a list of stocks and their covariance matrix.
#### Problem: 
   Formulates the optimization problem with constraints and various optimization methods.
#### Key Methods:
  evaluator(solution): Validates the solution based on constraints.
  
  solver_cplex(): Solves the problem using CPLEX and Pyomo.
  
  answer_greedy(): Provides a greedy solution for the problem.
  
  cost(solution), cost_2(current_solution, new_solution, old_cost), cost_3(solution): Functions to compute      different cost metrics.
  
  local_search_n_one(), local_search_n_two(), local_search_n_clear(): Local search strategies to improve        solutions.
  
  VNS(): Variable Neighborhood Search method for optimization.
  
  tabu_search(): Tabu Search method for finding optimal solutions.
  
  calculate_risk(solution), calculate_profit(solution): Calculates the risk and profit of a given solution.

## chart.py
### Description: 
Generates a plot to visualize the effect of different lambda values on the risk/return trade-off.

### Usage:
- Reads stock data and covariance matrix.
- Computes the risk/return ratio for various lambda values.
- Plots the results to show how lambda influences the risk/return balance.

## Stock_Generator.py
### Description: 
Provides functions to generate and manage stock data and covariance matrices.

### Key Functions:
  generate_variance(expected_return, number_of_choices=10): Generates a variance based on expected return.
  
  generate_stock_data(filename): Generates and saves random stock data to a file.
  
  create_stocks_from_txt(filename): Reads stock data from a file and creates Stock objects.
  
  create_stock(): Generates and returns a list of stocks.
  
  read_stock(): Reads stock data from a file and returns a list of Stock objects.
  
## Feel free to modify the datas in StockData.txt and MatrixData.txt...
