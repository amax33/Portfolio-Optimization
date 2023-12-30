import numpy as np
import pyomo.environ
from pyomo.environ import *


class Stock:
    def __init__(self, expected_return, variance, cost):
        self.expected_return = expected_return  # Profit
        self.variance = variance  # Risk
        self.cost = cost  # Cost


class Market:
    def __init__(self, stocks, covariance):
        self.stocks = stocks  # List of Stocks in the Market
        self.covariance = covariance
        # 2D array consists of covariance between two stock i & j

    def get_number_of_stocks(self):
        return len(self.stocks)


class Problem:
    def __init__(self, budget, max_variance, max_variance_total, market):
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
                summation += solution[i] * solution[j] * self.market.covariance[i][j]

        if summation > self.max_variance_total:
            error[3] = 1

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
        model.obj = Objective(expr=sum(model.r[i] * model.x[i] for i in model.assets), sense=maximize)

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

        return model.x


if __name__ == "__main__":
    # Example usage
    stock1 = Stock(expected_return=0.05, variance=0.005, cost=150)
    stock2 = Stock(expected_return=0.1, variance=0.01, cost=100)

    market = Market(stocks=[stock1, stock2], covariance=[[0, 0.0005], [0.0005, 0]])

    problem = Problem(budget=200, max_variance=0.02, max_variance_total=0.03, market=market)

    answer_cplex = problem.solver_cplex()
    answer = np.zeros(problem.market.get_number_of_stocks())

    for i in range(problem.market.get_number_of_stocks()):
        answer[i] = answer_cplex[i].value

    # checking the answer with the evaluator function
    constrain = problem.evaluator(answer)
    for i in range(problem.market.get_number_of_stocks()):
        if constrain[i] != 0:
            print('Constrain number ' + str(i) + ' has not comply!')

    for i in range(problem.market.get_number_of_stocks()):
        if answer[i] == 1:
            print('Invest in asset ', i + 1)
