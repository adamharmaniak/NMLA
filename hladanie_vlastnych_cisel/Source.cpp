#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

Vec matVecMul(const Mat& A, const Vec& x) {
    int n = A.size();
    Vec result(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result[i] += A[i][j] * x[j];
    return result;
}

double maxAbs(const Vec& x) {
    double max_val = 0.0;
    for (double val : x)
        max_val = max(max_val, fabs(val));
    return max_val;
}

double norm(const Vec& x) {
    double sum = 0.0;
    for (double val : x)
        sum += val * val;
    return sqrt(sum);
}

void powerMethod(const Mat& A, Vec x0, int maxit, double tol) {
    int k = 0;
    Vec x = x0;
    double lambda_m = 0.0;

    while (k < maxit) {
        ++k;
        Vec x_tilde = matVecMul(A, x);
        lambda_m = maxAbs(x_tilde);
        for (int i = 0; i < x.size(); i++)
            x[i] = x_tilde[i] / lambda_m;
        Vec r = matVecMul(A, x);
        for (int i = 0; i < x.size(); i++)
            r[i] -= lambda_m * x[i];
        if (norm(r) < tol)
            break;
    }

    cout << "Najvacsie vlastne cislo (power method): " << lambda_m << endl;
}

Vec solveLinearSystem(Mat A, Vec b) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        A[i].push_back(b[i]);
    }

    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int j = i + 1; j < n; j++)
            if (fabs(A[j][i]) > fabs(A[pivot][i]))
                pivot = j;
        swap(A[i], A[pivot]);

        for (int j = i + 1; j < n; j++) {
            double ratio = A[j][i] / A[i][i];
            for (int k = i; k <= n; k++)
                A[j][k] -= ratio * A[i][k];
        }
    }

    Vec x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = A[i][n];
        for (int j = i + 1; j < n; j++)
            x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }

    return x;
}

void inverseIteration(const Mat& A, Vec x0, double sigma, int maxit, double eps) {
    int n = A.size();
    Vec x = x0;
    double lambda_approx = 0.0;

    for (int k = 0; k < maxit; k++) {
        Mat B = A;
        for (int i = 0; i < n; i++)
            B[i][i] -= sigma;

        Vec y = solveLinearSystem(B, x);

        double norm_y = norm(y);
        for (int i = 0; i < n; i++)
            x[i] = y[i] / norm_y;

        Vec Ax = matVecMul(A, x);
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 0; i < n; i++) {
            numerator += x[i] * Ax[i];
            denominator += x[i] * x[i];
        }
        lambda_approx = numerator / denominator;

        Vec r(n);
        for (int i = 0; i < n; i++)
            r[i] = Ax[i] - lambda_approx * x[i];

        if (norm(r) < eps)
            break;
    }

    cout << "Najmensie vlastne cislo (inverse iteration): " << lambda_approx << endl;
}

Mat loadMatrixFromFile(const string& filename, int n) {
    Mat A(n, Vec(n));
    ifstream file(filename);
    if (!file) {
        cerr << "Nepodarilo sa otvorit subor " << filename << endl;
        exit(1);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> A[i][j];
    return A;
}

double rayleighQuotient(const Mat& A, const Vec& x) {
    Vec Ax = matVecMul(A, x);
    double numerator = 0.0, denominator = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        numerator += x[i] * Ax[i];
        denominator += x[i] * x[i];
    }
    return numerator / denominator;
}

void deflateMatrix(Mat& A, const Vec& v, double lambda) {
    int n = A.size();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] -= lambda * v[i] * v[j];
}

pair<double, Vec> findEigenpair(Mat A, Vec x0, int maxit, double tol) {
    Vec x = x0;
    for (int k = 0; k < maxit; k++) {
        Vec x_tilde = matVecMul(A, x);
        double norm_x = norm(x_tilde);
        for (int i = 0; i < x.size(); i++)
            x[i] = x_tilde[i] / norm_x;

        Vec Ax = matVecMul(A, x);
        Vec r(x.size());
        for (int i = 0; i < x.size(); i++)
            r[i] = Ax[i] - rayleighQuotient(A, x) * x[i];

        if (norm(r) < tol)
            break;
    }

    double lambda = rayleighQuotient(A, x);
    return { lambda, x };
}

void orthogonalize(Vec& x, const vector<Vec>& prev_vectors) {
    for (const auto& v : prev_vectors) {
        double dot_prod = 0.0;
        for (int i = 0; i < x.size(); i++)
            dot_prod += x[i] * v[i];
        for (int i = 0; i < x.size(); i++)
            x[i] -= dot_prod * v[i];
    }
}

int main() {
    srand(time(0));
    int n = 9;
    Mat A = loadMatrixFromFile("matica.txt", n);

    Vec x0(n);
    for (int i = 0; i < n; ++i) x0[i] = rand() / (double)RAND_MAX;

    pair<double, Vec> eig1 = findEigenpair(A, x0, 1000, 1e-6);
    double lambda1 = eig1.first;
    Vec v1 = eig1.second;

    deflateMatrix(A, v1, lambda1);

    Vec x1(n);
    for (int i = 0; i < n; ++i) x1[i] = rand() / (double)RAND_MAX;
    orthogonalize(x1, { v1 });
    pair<double, Vec> eig2 = findEigenpair(A, x1, 1000, 1e-6);
    double lambda2 = eig2.first;
    Vec v2 = eig2.second;

    deflateMatrix(A, v2, lambda2);

    Vec x2(n);
    for (int i = 0; i < n; ++i) x2[i] = rand() / (double)RAND_MAX;
    orthogonalize(x2, { v1, v2 });
    pair<double, Vec> eig3 = findEigenpair(A, x2, 1000, 1e-6);
    double lambda3 = eig3.first;
    Vec v3 = eig3.second;

    cout << "1. vlastne cislo: " << lambda1 << endl;
    cout << "2. vlastne cislo: " << lambda2 << endl;
    cout << "3. vlastne cislo: " << lambda3 << endl;

    return 0;
}
