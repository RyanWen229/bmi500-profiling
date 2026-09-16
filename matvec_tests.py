import unittest

from matvec_multiply import dot_product, matrix_vector_product


class TestMatrixVectorMultiply(unittest.TestCase):
    def test_dot_product(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)

    def test_matrix_vector_product(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        vector = [1, 0, -1]
        self.assertEqual(matrix_vector_product(matrix, vector), [-2, -2, -2])


if __name__ == "__main__":
    unittest.main()
