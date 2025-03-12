#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

double dotProduct(const Vector& a, const Vector& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

Vector scalarMultiply(double scalar, const Vector& v) {
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = scalar * v[i];
    }
    return result;
}

Vector vectorSubtract(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double norm(const Vector& v) {
    return sqrt(dotProduct(v, v));
}

double normMatrix(const Matrix& A) {
    double sum = 0.0;
    for (const auto& row : A) {
        for (double val : row) {
            sum += val * val;
        }
    }
    return sqrt(sum);
}

Matrix multiplyMatrices(const Matrix& A, const Matrix& B) {
    size_t n = A.size();
    Matrix C(n, Vector(n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix transposeMatrix(const Matrix& A) {
    size_t n = A.size();
    size_t m = A[0].size();
    Matrix T(m, Vector(n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Matrix subtractIdentity(const Matrix& A) {
    size_t n = A.size();
    Matrix I(n, Vector(n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            I[i][j] = (i == j) ? 1.0 : 0.0;
        }
        for (size_t j = 0; j < n; j++) {
            I[i][j] = A[i][j] - I[i][j];
        }
    }
    return I;
}

void gramSchmidt(const Matrix& A, Matrix& Q, Matrix& R) {
    size_t k = A.size();
    size_t n = A[0].size();
    Q.assign(k, Vector(n, 0.0));
    R.assign(k, Vector(k, 0.0));

    for (size_t j = 0; j < k; j++) {
        Vector v = A[j];
        for (size_t i = 0; i < j; i++) {
            R[i][j] = dotProduct(Q[i], A[j]);
            v = vectorSubtract(v, scalarMultiply(R[i][j], Q[i]));
        }
        R[j][j] = norm(v);
        if (R[j][j] > 1e-10) {
            Q[j] = scalarMultiply(1.0 / R[j][j], v);
        }
    }
}

void gramSchmidtModified(const Matrix& A, Matrix& Q, Matrix& R) {
    size_t k = A.size();
    size_t n = A[0].size();
    Q.assign(k, Vector(n, 0.0));
    R.assign(k, Vector(k, 0.0));
    Matrix V = A;

    for (size_t i = 0; i < k; i++) {
        R[i][i] = norm(V[i]);
        if (R[i][i] > 1e-10) {
            Q[i] = scalarMultiply(1.0 / R[i][i], V[i]);
        }
        for (size_t j = i + 1; j < k; j++) {
            R[i][j] = dotProduct(Q[i], V[j]);
            V[j] = vectorSubtract(V[j], scalarMultiply(R[i][j], Q[i]));
        }
    }
}

void printMatrix(const Matrix& M) {
    for (const auto& row : M) {
        std::cout << "| ";
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "|" << std::endl;
    }
}

Matrix generateMatrix(int numRows, int numCols) {
    srand(time(0));
    Matrix A(numRows, Vector(numCols));
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            A[i][j] = (rand() % 200 - 100) / 10.0;
        }
    }
    return A;
}

int main() {
    Matrix A = generateMatrix(200, 200);

    Matrix Q1, Q2, R;
    gramSchmidt(A, Q1, R);
    gramSchmidtModified(A, Q2, R);

    /*std::cout << "Povodna matica: " << std::endl;
    printMatrix(A);
    std::cout << "Ortonormalna matica podla Gram-Schmidtovho procesu:\n\n";
    printMatrix(Q1);
    std::cout << "Ortonormalna matica podla modifikovaneho Gram-Schmidtovho procesu:\n\n";
    printMatrix(Q2);*/

    Matrix Q1_transposed = transposeMatrix(Q1);
    Matrix Q1Q1T = multiplyMatrices(Q1, Q1_transposed);
    Matrix diff1 = subtractIdentity(Q1Q1T);
    double norm1 = normMatrix(diff1);

    Matrix Q2_transposed = transposeMatrix(Q2);
    Matrix Q2Q2T = multiplyMatrices(Q2, Q2_transposed);
    Matrix diff2 = subtractIdentity(Q2Q2T);
    double norm2 = normMatrix(diff2);

    std::cout << "||(Q1 * Q1^T) - I||: " << norm1 << std::endl;
    std::cout << "||(Q2 * Q2^T) - I||: " << norm2 << std::endl;

    return 0;
}
