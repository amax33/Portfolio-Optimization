import numpy as np
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
        summation = 0
        for i in range(self.Num):
            for j in range(self.Num):
                if solution[i] == solution[j] == 1:
                    risk = solution[i] * solution[j] * self.market.covariance[i][j]
                    if summation < risk:
                        summation = risk

        if summation > self.max_variance_total:
            error[3] = 1

        return error

    def answer_greedy(self):
        for i, stock in enumerate(self.market.stocks):
            stock.score = stock.expected_return * stock.cost #/ stock.variance
            stock.number = i

        solution = np.zeros(self.Num)
        budget = 0
        stock_choose = copy.deepcopy(self.market.stocks)
        while len(stock_choose) > 0:
            max_score_stock = max(stock_choose, key=lambda x: x.score)
            if max_score_stock.cost + budget <= self.Budget:
                solution[max_score_stock.number] = 1
                # if np.any(self.evaluator(solution)):
                #     solution[max_score_stock.number] = 0
                #     stock_choose.remove(max_score_stock)
                #     continue
                budget += max_score_stock.cost
            stock_choose.remove(max_score_stock)
            if budget > self.Budget:
                break
        return solution

    def cost(self, solution):
        if np.any(self.evaluator(solution)):
            return np.inf  # non-acceptable solution

        total_variance = 0
        total_return = 0
        for i in range(self.Num):
            for j in range(self.Num):
                total_variance += solution[i] * solution[j] * self.market.covariance[i][j]/100
            total_return += solution[i] * self.market.stocks[i].expected_return * self.market.stocks[i].cost/self.Budget
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
                        range(self.Num) for k in range(self.Num)) / 100
                else:  # if the stock was not present in the current solution and has been added
                    delta_return += self.market.stocks[i].expected_return * self.market.stocks[i].cost
                    delta_variance += sum(
                        new_solution[j] * new_solution[k] * self.market.covariance[j][k] for j in range(self.Num) for k
                        in range(self.Num)) / 100

        new_cost = old_cost + delta_variance - self.lambda_ * delta_return

        return new_cost

    def local_search_n_one(self):
        current_solution = self.answer_greedy()
        cost = self.cost(current_solution)
        best_neighbor = current_solution
        best_cost = cost
        no_improvement_iterations = 0
        for i in range(self.Num*2):
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

    def local_search_n_rand(self):
        current_solution = self.answer_greedy()
        cost = self.cost(current_solution)
        best_neighbor = current_solution
        best_cost = cost
        no_improvement_iterations = 0
        neighborhood_size = self.Num // 100

        for i in range(self.Num*2):
            if no_improvement_iterations > 100:
                neighborhood_size += 1  # Increase neighborhood size
            if no_improvement_iterations > self.Num:
                break
            neighbor = copy.deepcopy(current_solution)
            for _ in range(neighborhood_size):
                rand = random.randint(0, self.Num - 1)
                neighbor[rand] = 1 - current_solution[rand]
            if cost == np.inf:
                cost = self.cost(neighbor)
            else:
                cost = self.cost_2(current_solution, neighbor, cost)
            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor
                no_improvement_iterations = 0
            else:
                no_improvement_iterations += 1
        return best_neighbor, best_cost

    def local_search_n_block(self):
        current_solution = self.answer_greedy()
        cost = self.cost(current_solution)
        best_neighbor = current_solution
        best_cost = cost
        no_improvement_iterations = 0
        neighborhood_size = self.Num // 100

        for i in range(self.Num*2):
            if no_improvement_iterations > 100:
                neighborhood_size = min(neighborhood_size + 1, self.Num)  # Ensure not exceeding bounds
            if no_improvement_iterations > self.Num:
                break
            neighbor = copy.deepcopy(current_solution)
            rand = random.randint(0, self.Num - 1)
            for j in range(neighborhood_size):
                neighbor[(j + rand) % self.Num] = 1 - current_solution[(j + rand) % self.Num]
            if cost == np.inf:
                cost = self.cost(neighbor)
            else:
                cost = self.cost_2(current_solution, neighbor, cost)
            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor
                no_improvement_iterations = 0
            else:
                no_improvement_iterations += 1

        return best_neighbor, best_cost

    def VNS(self):
        # Initial solution
        current_solution = self.answer_greedy()
        current_cost = self.cost(current_solution)
        improvement = 0
        # Define the local search methods
        local_search_methods = [self.local_search_n_one, self.local_search_n_rand, self.local_search_n_block]
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
    problem = Problem(budget=400, max_variance=0.1, max_variance_total=0.5, market=market)
    answer, cost = problem.VNS()



    for i in range(problem.market.get_number_of_stocks()):
        if answer[i] >= 1:
            print('Invest in asset ', i + 1)
    print(cost)

