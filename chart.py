import Stock_Generator
import calculator
import matplotlib.pyplot as plt

Ans = list()
X = [0.1, 0.5, 1]
stocks = Stock_Generator.read_stock()
covariance = Stock_Generator.read_matrix_from_file('MatrixData.txt')
market = calculator.Market(stocks, covariance)
for x in X:
    problem = calculator.Problem(budget=400, max_variance=0.05, max_variance_total=0.5, market=market, lambda_=x)
    answer, cost = problem.VNS()
    Ans.append(abs(problem.calculate_risk(answer)/problem.calculate_profit(answer))*100)
plt.plot(X, Ans)
plt.xlabel("lambda value")
plt.show()
