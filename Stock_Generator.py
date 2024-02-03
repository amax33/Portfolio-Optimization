import random
from main import Stock
import math

global stocks
stocks = []
number = 500

def generate_variance(expected_return, number_of_choices = 10):
    variances = []
    for _ in range(number_of_choices):
        alpha = random.uniform(1.00,6.00)
        generated_var = math.exp(-alpha * expected_return) / 10
        variances.append(generated_var)
    selected_var = random.choice(variances)
    return selected_var

def generate_stock_data(filename):
    with open(filename, 'w') as f:
        f.truncate()
        for _ in range(number):
            expected_return = random.uniform(0.05, 0.30)
            variance = generate_variance(expected_return)
            cost = random.randint(25, 70)
            data = f"{expected_return},{variance},{cost}\n"
            f.write(data)


def create_stocks_from_txt(filename):
    with open(filename) as f:
        lines = f.readlines()

    for _ in range(number):
        # Split the line into a list of values
        values = lines[_].strip().split(',')
        # Create a new Stock object
        stock = Stock(
            expected_return=float(values[0]),
            variance=float(values[1]),
            cost=float(values[2])
        )
        stocks.append(stock)


def create_stock():
    generate_stock_data('StockData.txt')
    create_stocks_from_txt('StockData.txt')
    return stocks

def read_stock():
    create_stocks_from_txt('StockData.txt')
    return stocks

def create_matrix():
    save_matrix_to_file(create_symmetric_matrix(), 'MatrixData.txt')
    return read_matrix_from_file('MatrixData.txt')


def create_symmetric_matrix():
    matrix = [[0] * number for _ in range(number)]  # Initialize a matrix with zeros

    for i in range(number):
        for j in range(i + 1, number):  # Loop only over the upper triangular part
            # Generate a random number between 0 and 9 (inclusive)
            if i != j:
                random_number = random.randint(-6, 6) / 10
                matrix[i][j] = random_number  # Assign random number to (i, j)
                matrix[j][i] = random_number  # Assign same random number to (j, i)
    return matrix


def save_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        file.truncate()
        for row in matrix:
            row_str = ' '.join(map(str, row))
            file.write(row_str + '\n')


def read_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            matrix.append(row)
    return matrix