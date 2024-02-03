import collections
import heapq

import numpy as np
import pyomo.environ
from pyomo.environ import *
import Stock_Generator
import copy
import random


class Stock:
    def __init__(self, expected_return, variance, cost):
        self.expected_return = expected_return  # Profit
        self.variance = variance  # Risk
        self.cost = cost  # Cost
        self.score = 0
        self.number = 0


class Market:
    def __init__(self, stocks, covariance):
        self.stocks = stocks  # List of Stocks in the Market
        self.covariance = covariance
        # 2D array consists of covariance between two stock i & j

    def get_number_of_stocks(self):
        return len(self.stocks)


class Problem:
    def __init__(self, budget, max_variance, max_variance_total, market, lambda_=1):
        self.lambda_ = lambda_  # trade-off between risk and return
        self.Budget = budget  # total budget of the investor
        self.max_variance = max_variance
        # maximum value for the risk of each asset
        self.max_variance_total = max_variance_total
        # maximum value for the risk of the total portfolio
        self.market = market  # market of the problem
        self.Num = self.market.get_number_of_stocks()

    def evaluator(self, solution):
        error = np.zeros(4)  # a list in size of the number of constraints

        # checking solution validation
        for i in range(self.Num):
            if solution[i] != 0 and solution[i] != 1:
                error[0] = 1

        # checking risk of each asset
        for i in range(self.Num):
            if solution[i] * self.market.stocks[i].variance > self.max_variance:
                error[1] = 1

        # checking budget capability
        summation = 0
        for i in range(self.Num):
            summation += solution[i] * self.market.stocks[i].cost
        if summation > self.Budget:
            error[2] = 1

        # checking total risk of portfolio
        # summation = 0
        # for i in range(self.Num):
        #     for j in range(self.Num):
        #         if solution[i] == solution[j] == 1:
        #             risk = solution[i] * solution[j] * self.market.covariance[i][j]
        #             if summation < risk:
        #                 summation = risk
        #
        # if summation > self.max_variance_total:
        #     error[3] = 1

        return error

    def solver_cplex(self):
        model = ConcreteModel()
        # Sets
        model.assets = Set(initialize=range(problem.market.get_number_of_stocks()))  # set of assets
        # Parameters
        model.r = Param(model.assets, initialize={i: problem.market.stocks[i].expected_return for i in
                                                  model.assets})  # expected return
        model.v = Param(model.assets,
                        initialize={i: problem.market.stocks[i].variance for i in model.assets})  # variance
        model.c = Param(model.assets, initialize={i: problem.market.stocks[i].cost for i in model.assets})  # cost
        model.sigma = Param(model.assets, model.assets,
                            initialize={(i, j): problem.market.covariance[i][j] for i in model.assets for j in
                                        model.assets})  # covariance
        # Variables
        model.x = Var(model.assets, within=pyomo.environ.Binary)  # binary variable for each asset
        # Objective function
        model.obj = Objective(expr=sum(model.r[i] * model.x[i] * model.c[i] for i in model.assets), sense=maximize)
        # Constraints
        model.budget_constraint = Constraint(expr=sum(model.c[i] * model.x[i] for i in model.assets) <= problem.Budget)
        model.risk_total_constraint = Constraint(expr=sum(
            model.x[i] * model.x[j] * model.sigma[i, j] for i in model.assets for j in
            model.assets) <= problem.max_variance_total)
        model.risk_each_constraint = ConstraintList()
        for i in model.assets:
            model.risk_each_constraint.add(model.v[i] * model.x[i] <= problem.max_variance)
        # Solver
        solver = SolverFactory('cplex')
        solver.solve(model)
        solution = np.array([model.x[i].value for i in range(self.Num)])
        return solution

    def answer_greedy(self):
        for i, stock in enumerate(self.market.stocks):
            stock.score = stock.expected_return * stock.cost #/ stock.variance
            stock.number = i

        solution = np.zeros(self.Num)
        budget = 0
        stock_choose = copy.deepcopy(self.market.stocks)
        while len(stock_choose) > 0:
            max_score_stock = max(stock_choose, key=lambda x: x.score)
            if max_score_stock.cost + budget <= self.Budget - max_score_stock.cost and max_score_stock.variance < self.max_variance:
                solution[max_score_stock.number] = 1
                budget += max_score_stock.cost
            stock_choose.remove(max_score_stock)
            if budget > self.Budget:
                break
        # print(self.calculate_risk(solution))
        # print(self.calculate_profit(solution))
        return solution

    def cost(self, solution):
        if np.any(self.evaluator(solution)):
            return np.inf  # non-acceptable solution

        total_variance = 0
        total_return = 0
        for i in range(self.Num):
            for j in range(self.Num):
                total_variance += solution[i] * solution[j] * self.market.covariance[i][j]/10
            total_return += solution[i] * self.market.stocks[i].expected_return * self.market.stocks[i].cost/self.Budget
            #print("P" + str(total_return) + "R" + str(total_variance))
        return total_variance - self.lambda_ * total_return

    def cost_2(self, current_solution, new_solution, old_cost):
        if np.any(self.evaluator(new_solution)):
            return np.inf  # non-acceptable solution

        delta_return = 0
        delta_variance = 0

        for i in range(self.Num):
            if current_solution[i] != new_solution[i]:  # if the stock has been added or removed
                if current_solution[i] == 1:  # if the stock was present in the current solution and has been removed
                    delta_return -= self.market.stocks[i].expected_return * self.market.stocks[i].cost
                    delta_variance -= sum(
                        current_solution[j] * current_solution[k] * self.market.covariance[j][k] for j in
                        range(self.Num) for k in range(self.Num))/10
                else:  # if the stock was not present in the current solution and has been added
                    delta_return += self.market.stocks[i].expected_return * self.market.stocks[i].cost
                    delta_variance += sum(
                        new_solution[j] * new_solution[k] * self.market.covariance[j][k] for j in range(self.Num) for k
                        in range(self.Num)) / 10

        new_cost = old_cost - delta_variance + self.lambda_ * delta_return

        return new_cost

    def cost_3(self, solution):
        if np.any(self.evaluator(solution)):
            return np.inf  # non-acceptable solution
        total_return = 0
        for i in range(self.Num):
            total_return += solution[i] * self.market.stocks[i].expected_return * self.market.stocks[i].cost/self.Budget
            #print("P" + str(total_return) + "R" + str(total_variance))
        return -total_return

    def local_search_n_one(self):
        current_solution = self.answer_greedy()
        cost = self.cost(current_solution)
        best_neighbor = current_solution
        best_cost = cost
        no_improvement_iterations = 0
        for i in range(self.Num*5):
            if no_improvement_iterations > self.Num:  # Stopping criteria based on lack of improvement
                break
            neighbor = copy.deepcopy(current_solution)
            neighbor[i % self.Num] = 1 - current_solution[i % self.Num]  # flip the bit
            if cost == np.inf:
                cost = self.cost(neighbor)
            else:
                cost = self.cost_2(current_solution, neighbor, cost)

            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor
                no_improvement_iterations = 0  # Reset counter on improvement
            else:
                no_improvement_iterations += 1
        return best_neighbor, best_cost

    def local_search_n_two(self):
        current_solution = self.answer_greedy()
        cost = self.cost(current_solution)
        best_neighbor = current_solution
        best_cost = cost

        for i in range(self.Num*5):
            neighbor = copy.deepcopy(current_solution)
            for _ in range(2):
                rand = random.randint(0, self.Num - 1)
                neighbor[rand] = 1 - current_solution[rand]
            if cost == np.inf:
                cost = self.cost(neighbor)
            else:
                cost = self.cost_2(current_solution, neighbor, cost)
            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor
        return best_neighbor, best_cost

    def local_search_n_clear(self):
        current_solution = self.answer_greedy()
        cost = self.cost(current_solution)
        best_neighbor = current_solution
        best_cost = cost
        for i in range(self.Num*2):
            neighbor = copy.deepcopy(current_solution)
            counter = 0
            for j in range(self.Num):
                if counter == 1:
                    break
                if neighbor[j] == 1:
                    neighbor[j] = 0
                    counter += 1
            for _ in range(2):
                rand = random.randint(0, self.Num - 1)
                neighbor[rand] = 1 - current_solution[rand]

            if cost == np.inf:
                cost = self.cost(neighbor)
            else:
                cost = self.cost_2(current_solution, neighbor, cost)
            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor

        return best_neighbor, best_cost

    def VNS(self):
        # Initial solution
        current_solution = self.answer_greedy()
        current_cost = self.cost(current_solution)
        improvement = 0
        # Define the local search methods
        local_search_methods = [self.local_search_n_one, self.local_search_n_two, self.local_search_n_clear]
        # Iteratively apply local search methods
        for local_search in local_search_methods:
            new_solution, new_cost = local_search()
            # If there's an improvement, update the current solution and cost
            if new_cost < current_cost:
                current_solution, current_cost = new_solution, new_cost
                improvement = 0
            else:
                improvement += 1
            if improvement > 3:
                break
        return current_solution, current_cost

    def tabu_search(self, iterations=100, tabu_tenure=5, num_neighbors=10):
        current_solution = self.answer_greedy()
        best_solution = current_solution
        best_cost = self.cost_3(current_solution)
        tabu_set = set()  # Use a set for the tabu list

        for it in range(iterations):
            neighborhood = []

            # Generate a limited number of random neighbors
            for _ in range(num_neighbors):
                neighbor = copy.deepcopy(current_solution)
                counter = 0
                for j in range(self.Num):
                    if counter == 1:
                        break
                    if neighbor[j] == 1:
                        neighbor[j] = 0
                        counter += 1
                for _ in range(2):
                    rand = random.randint(0, self.Num - 1)
                    neighbor[rand] = 1 - current_solution[rand]

                # Check if neighbor is tabu (simplified check assuming unique solutions)
                if tuple(neighbor) not in tabu_set and self.cost_3(neighbor) != np.inf:
                    if best_cost != np.inf:
                        heapq.heappush(neighborhood, (self.cost_2(best_solution, neighbor, best_cost), neighbor))  # Use a heap for the neighborhood
                    else:
                        heapq.heappush(neighborhood, (self.cost_3(neighbor), neighbor))

            # If no valid neighbors, continue to next iteration
            if not neighborhood:
                continue

            # Find the best neighbor
            best_neighbor_cost, best_neighbor = heapq.heappop(
                neighborhood)  # Use heap operation to find the minimum element

            # Update if the best neighbor is better than the best solution found so far
            if best_neighbor_cost < best_cost:
                best_solution = best_neighbor
                best_cost = best_neighbor_cost
            tabu_set.add(tuple(current_solution))  # Add current solution to tabu set
            if len(tabu_set) > tabu_tenure:
                tabu_set.pop()  # Remove the oldest element from the tabu set
            current_solution = best_neighbor

        return best_solution, best_cost

    def calculate_risk(self, solution):
        T_risk = 0
        for i in range(self.Num):
            T_risk += solution[i] * self.market.stocks[i].variance
            for j in range(self.Num):
                T_risk += solution[i] * solution[j] * self.market.covariance[i][j]

        return T_risk

    def calculate_profit(self, solution):
        T_profit = 0
        for i in range(self.Num):
            T_profit += solution[i] * self.market.stocks[i].expected_return
        return T_profit




if __name__ == "__main__":
    num = int(input("Condition:"))
    # Example usage
    if num == 0:
        stocks = Stock_Generator.create_stock()
        covariance = Stock_Generator.create_matrix()
    else:
        stocks = Stock_Generator.read_stock()
        covariance = Stock_Generator.read_matrix_from_file('MatrixData.txt')

    market = Market(stocks, covariance)
    problem = Problem(budget=400, max_variance=0.05, max_variance_total=0.05, market=market)
    answer, cost = problem.VNS()
    for i in range(problem.market.get_number_of_stocks()):
        if answer[i] >= 1:
            print('Invest in asset ', i + 1)
    #print(cost)
    print(problem.calculate_risk(answer))
    print(problem.calculate_profit(answer))


    # Use the Pyomo-based solution
    answer_pyomo = problem.solver_cplex()
    # Print the Pyomo-based results
    for i in range(problem.market.get_number_of_stocks()):
        if answer_pyomo[i] >= 1:
            print('Invest in asset ', i + 1)

    print(problem.calculate_risk(answer_pyomo))
    print(problem.calculate_profit(answer_pyomo))





