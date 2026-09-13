import random

def dot_product(vector_a, vector_b):
    """Return the dot product of two vectors."""
    total = 0
    for index in range(len(vector_a)):
        total += vector_a[index] * vector_b[index]
    return total


def matrix_vector_product(matrix, vector):
    """Return the product of a rectangular matrix and a vector."""
    vector_length = len(vector)
    result = []
    for row in matrix:
        if len(row) != vector_length:
            raise ValueError("each matrix row must have the same length as the vector")
        result.append(dot_product(row, vector))
    return result


def main():
    """Exercise matrix-vector multiplication with 1000 by 1000 random data."""
    size = 1000
    matrix = [[random.random() for _ in range(size)] for _ in range(size)]
    vector = [random.random() for _ in range(size)]
    return matrix_vector_product(matrix, vector)


if __name__ == "__main__":
    main()
