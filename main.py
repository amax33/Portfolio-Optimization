import numpy as np
import Stock_Generator
import copy


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
                risk = solution[i] * solution[j] * self.market.covariance[i][j]
                if summation < risk:
                    summation = risk

        if summation > self.max_variance_total:
            error[3] = 1

        return error

    def cost_greedy(self):
        solution = np.zeros(self.Num)
        for i, stock in enumerate(self.market.stocks):
            stock.score = stock.expected_return * stock.cost #/ stock.variance
            stock.number = i

        sorted_stock = sorted(self.market.stocks, key=lambda x: x.score, reverse=False)
        solution = np.zeros(self.Num)
        Budget = 0
        for i in range(self.Num):
            Budget += sorted_stock[i].cost
            if Budget <= self.Budget and self.market.stocks[sorted_stock[i].number].variance < self.max_variance:
                solution[sorted_stock[i].number] = 1
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

        print("Va", total_variance, "Ri", total_return)
        return total_variance - self.lambda_ * total_return

    def cost_2(self, current_solution, new_solution, current_cost):
        if np.any(self.evaluator(new_solution)):
            return np.inf  # non-acceptable solution

        delta_return = 0
        delta_variance = 0
        for i in range(self.Num):
            if current_solution[i] != new_solution[i]:  # if the stock has been added or removed
                if current_solution[i] == 1:
                    delta_return -= self.market.stocks[i].expected_return * self.market.stocks[i].cost
                else:
                    delta_return += self.market.stocks[i].expected_return * self.market.stocks[i].cost

                for j in range(self.Num):
                    risk = new_solution[j] * self.market.covariance[i][j]
                    if delta_variance < risk:
                        delta_variance = risk

        print("cost", delta_return)
        print("v", delta_variance)
        # if we removed the stock
        if np.sum(new_solution) < np.sum(current_solution):
            total_return = current_cost - self.lambda_ * delta_return
            total_variance = current_cost - delta_variance
        else:  # if we added the stock
            total_return = current_cost + self.lambda_ * delta_return
            total_variance = current_cost + delta_variance

        return total_variance - self.lambda_ * total_return

    def n_one(self, solution):
        neighbors = []
        for i in range(self.Num):
            neighbor = copy.deepcopy(solution)
            neighbor[i] = 1 - neighbor[i]  # flip the bit
            neighbors.append(neighbor)

        return neighbors

    def local_search(self):
        current_solution = self.cost_greedy()
        best_neighbor = current_solution
        best_cost = np.inf
        for i in range(100):
            neighbor = copy.deepcopy(current_solution)
            neighbor[i%self.Num] = 1 - neighbor[i%self.Num]  # flip the bit
            cost = problem.cost(neighbor)
            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor
        current_solution = best_neighbor

        return current_solution, best_cost



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
    answer, cost = problem.local_search()
    print(answer)

    for i in range(problem.market.get_number_of_stocks()):
        if answer[i] >= 1:
            print('Invest in asset ', i + 1)
