import random
from main import Stock

global stocks
stocks = []


def generate_stock_data(filename, n):
    with open(filename, 'w') as f:
        f.truncate()
        for _ in range(n):
            expected_return = random.uniform(0.05, 0.30)
            variance = random.uniform(0.01, 0.10)
            cost = random.randint(25, 70)

            data = f"{expected_return},{variance},{cost}\n"
            f.write(data)


def create_stocks_from_txt(filename, n):
    with open(filename) as f:
        lines = f.readlines()

    for _ in range(n):
        # Split the line into a list of values
        values = lines[_].strip().split(',')
        # Create a new Stock object
        stock = Stock(
            expected_return=float(values[0]),
            variance=float(values[1]),
            cost=float(values[2])
        )
        stocks.append(stock)


def create():
    generate_stock_data('data.txt', 100)
    create_stocks_from_txt('data.txt', 100)
    return stocks


def create_symmetric_matrix(n):
    matrix = [[0] * n for _ in range(n)]  # Initialize a matrix with zeros

    for i in range(n):
        for j in range(i + 1, n):  # Loop only over the upper triangular part
            # Generate a random number between 0 and 9 (inclusive)
            if i != j:
                random_number = random.randint(0, 9)
                matrix[i][j] = random_number  # Assign random number to (i, j)
                matrix[j][i] = random_number  # Assign same random number to (j, i)

    return matrix
