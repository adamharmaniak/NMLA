#include<stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <sstream>

using namespace std;
typedef vector<double> Vector;
typedef vector<vector<double>> Matrix;

struct MatCRS {
    vector<double> val;
    vector<size_t> col_ind;
    vector<size_t> row_ptr;
    size_t n;
};

double dot(const Vector& a, const Vector& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

vector<double> matvec(const Matrix& A, const Vector& x) {
    size_t n = A.size();
    Vector result(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < x.size(); ++j)
            result[i] += A[i][j] * x[j];
    return result;
}

vector<double> matvec_crs(const MatCRS& A, const vector<double>& x) {
    vector<double> result(A.n, 0.0);
    for (size_t i = 0; i < A.n; ++i) {
        for (size_t idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
            result[i] += A.val[idx] * x[A.col_ind[idx]];
        }
    }
    return result;
}

Vector matTvec(const Matrix& A, const Vector& x) {
    size_t n = A.size();
    Vector result(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            result[i] += A[j][i] * x[j];
    return result;
}

Vector vec_add(const Vector& a, const Vector& b, double alpha) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + alpha * b[i];
    return result;
}

Vector vec_sub(const Vector& a, const Vector& b, double alpha = 1.0) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] - alpha * b[i];
    return result;
}

Vector mat_vec_mult(const Matrix& A, const Vector& x) {
    Vector result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[i].size(); j++)
            result[i] += A[i][j] * x[j];
    return result;
}

Vector mat_vec_mult2(const MatCRS& A, const Vector& x) {
    Vector y(A.n, 0.0);
    for (size_t i = 0; i < A.n; i++) {
        for (size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            y[i] += A.val[j] * x[A.col_ind[j]];
        }
    }
    return y;
}

double norm(const Vector& a) {
    return sqrt(dot(a, a));
}

Matrix load_matrix(const string& filename, int& n) {
    ifstream infile(filename);
    if (!infile.is_open())
        throw runtime_error("Nepodarilo sa otvorit subor.");

    vector<double> values;
    double value;
    while (infile >> value)
        values.push_back(value);

    n = static_cast<int>(sqrt(values.size()));
    if (n * n != static_cast<int>(values.size()))
        throw runtime_error("Pocet prvkov v subore nezodpoveda stvorcovej matici.");

    Matrix A(n, Vector(n));
    for (int i = 0; i < n * n; ++i)
        A[i / n][i % n] = values[i];

    return A;
}

MatCRS load_matrix_crs(const string& filepath, int n) {
    MatCRS A;
    A.n = n;

    ifstream file(filepath);
    if (!file) {
        cerr << "Subor sa nepodarilo otvorit.\n";
        exit(1);
    }

    A.row_ptr.push_back(0);
    for (int i = 0; i < n; i++) {
        int row_nnz = 0;
        for (int j = 0; j < n; j++) {
            double value;
            file >> value;
            if (value != 0.0) {
                A.val.push_back(value);
                A.col_ind.push_back(j);
                row_nnz++;
            }
        }
        A.row_ptr.push_back(A.row_ptr.back() + row_nnz);
    }

    file.close();
    return A;
}

Vector vec_comb(const Vector& a, const Vector& b, const Vector& c, double alpha, double omega) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + alpha * b[i] + omega * c[i];
    return result;
}

void BiCG(const Matrix& A, const Vector& b, Vector& x, int max_iter, double tol) {
    int n = b.size();
    Vector r = vec_sub(b, matvec(A, x), 1.0);
    Vector r_star = r;
    Vector p = r;
    Vector p_star = r_star;

    double r_star_dot_r = dot(r_star, r);

    cout << "BiCG: iteracia vs reziduum\n";

    for (int j = 0; j < max_iter; ++j) {
        Vector Ap = matvec(A, p);
        Vector ATp_star = matTvec(A, p_star);

        double alpha = r_star_dot_r / dot(p_star, Ap);

        x = vec_add(x, p, alpha);
        Vector r_new = vec_sub(r, Ap, alpha);
        Vector r_star_new = vec_sub(r_star, ATp_star, alpha);

        double res_norm = sqrt(dot(r_new, r_new));
        cout << j + 1 << " " << res_norm << "\n";

        if (res_norm < tol) {
            cout << "Konvergencia po " << j + 1 << " iteraciach.\n";
            return;
        }

        double r_star_dot_r_new = dot(r_star_new, r_new);
        double beta = r_star_dot_r_new / r_star_dot_r;
        r_star_dot_r = r_star_dot_r_new;

        r = r_new;
        r_star = r_star_new;
        p = vec_add(r, p, beta);
        p_star = vec_add(r_star, p_star, beta);
    }

    cout << "Nedosiahla sa konvergencia po " << max_iter << " iteraciach.\n";
}

void BiCGStab(const Matrix& A, const Vector& b, Vector& x, int max_iter, double tol) {
    int n = b.size();
    Vector r = vec_sub(b, matvec(A, x), 1.0);
    Vector r0_hat = r;
    Vector p = r;

    double alpha = 1.0, omega = 1.0, rho = 1.0, rho_prev = 1.0;

    Vector v(n, 0.0), s(n), t(n);

    cout << "BiCGSTAB: iteracia vs reziduum\n";

    for (int j = 0; j < max_iter; ++j) {
        rho = dot(r0_hat, r);
        if (rho == 0.0) break;

        if (j == 0) {
            p = r;
        }
        else {
            double beta = (rho / rho_prev) * (alpha / omega);
            for (int i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        v = matvec(A, p);
        alpha = rho / dot(r0_hat, v);

        s = vec_sub(r, v, alpha);
        t = matvec(A, s);
        omega = dot(t, s) / dot(t, t);

        x = vec_comb(x, p, s, alpha, omega);
        r = vec_sub(s, t, omega);

        double res_norm = sqrt(dot(r, r));
        cout << j + 1 << " " << res_norm << "\n";

        if (res_norm < tol) {
            cout << "Konvergencia po " << j + 1 << " iteraciach.\n";
            return;
        }

        rho_prev = rho;
    }

    cout << "Nedosiahla sa konvergencia po " << max_iter << " iteraciach.\n";
}

void BiCGStabCRS(const MatCRS& A, const vector<double>& b, vector<double>& x, int max_iter, double tol) {
    int n = A.n;
    vector<double> r = vec_sub(b, matvec_crs(A, x));
    vector<double> r0_hat = r;
    vector<double> p = r;

    double alpha = 1.0, omega = 1.0, rho = 1.0, rho_prev = 1.0;

    vector<double> v(n, 0.0), s(n), t(n);

    cout << "BiCGSTAB (CRS): iteracia vs reziduum\n";

    for (int j = 0; j < max_iter; ++j) {
        rho = dot(r0_hat, r);
        if (rho == 0.0) break;

        if (j == 0) {
            p = r;
        }
        else {
            double beta = (rho / rho_prev) * (alpha / omega);
            for (int i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        v = matvec_crs(A, p);
        alpha = rho / dot(r0_hat, v);

        s = vec_sub(r, v, alpha);
        t = matvec_crs(A, s);
        omega = dot(t, s) / dot(t, t);

        x = vec_comb(x, p, s, alpha, omega);
        r = vec_sub(s, t, omega);

        double res_norm = norm(r);
        cout << j + 1 << " " << res_norm << "\n";

        if (res_norm < tol) {
            cout << "Konvergencia po " << j + 1 << " iteraciach.\n";
            return;
        }

        rho_prev = rho;
    }

    cout << "Nedosiahla sa konvergencia po " << max_iter << " iteraciach.\n";
}

Vector conjugate_gradient(const Matrix& A, const Vector& b, Vector& x, double tol) {
    Vector residuals;
    Vector r = vec_add(b, mat_vec_mult(A, x), -1.0);  // r = b - Ax
    Vector p = r;
    int iter = 0;
    const int maxiter = 1000;

    cout << "CG: iteracia vs reziduum\n";

    while (norm(r) >= tol && iter <= maxiter) {
        Vector Ap = mat_vec_mult(A, p);
        double alpha = dot(r, r) / dot(p, Ap);
        x = vec_add(x, p, alpha);
        Vector r_new = vec_add(r, Ap, -alpha);
        double beta = dot(r_new, r_new) / dot(r, r);
        p = vec_add(r_new, p, beta);
        r = r_new;

        double res_norm = norm(r);
        residuals.push_back(res_norm);
        cout << iter + 1 << " " << res_norm << "\n";

        iter++;
    }

    if (norm(r) < tol) {
        cout << "Konvergencia po " << iter << " iteraciach.\n";
    }
    else {
        cout << "Nedosiahla sa konvergencia po " << maxiter << " iteraciach.\n";
    }

    return residuals;
}

Vector gradient_descent(const Matrix& A, const Vector& b, Vector& x, double tol) {
    Vector residuals;
    Vector r = vec_add(b, mat_vec_mult(A, x), -1.0);
    int iter = 0;
    const int maxiter = 1000;

    cout << "GD: iteracia vs reziduum\n";

    while (norm(r) >= tol && iter <= maxiter) {
        double alpha = dot(r, r) / dot(r, mat_vec_mult(A, r));
        x = vec_add(x, r, alpha);
        r = vec_add(b, mat_vec_mult(A, x), -1.0);
        double res_norm = norm(r);
        residuals.push_back(res_norm);
        cout << iter + 1 << " " << res_norm << "\n";

        iter++;
    }

    if (norm(r) < tol) {
        cout << "Konvergencia po " << iter << " iteraciach.\n";
    }
    else {
        cout << "Nedosiahla sa konvergencia po " << maxiter << " iteraciach.\n";
    }

    return residuals;
}

int detect_matrix_size(const string& filename) {
    ifstream infile(filename);
    if (!infile)
        throw runtime_error("Nepodarilo sa otvorit subor na detekciu velkosti.");

    int count = 0;
    double value;
    while (infile >> value) count++;

    int n = static_cast<int>(sqrt(count));
    if (n * n != count)
        throw runtime_error("Subor neobsahuje stvorcovu maticu.");
    return n;
}

int main() {
    try {
        int n1 = detect_matrix_size("C:\\Users\\student\\Harmaniak\\NMLA\\Solvers_cv8\\maticaCG.txt");
        int n2 = detect_matrix_size("C:\\Users\\student\\Harmaniak\\NMLA\\Solvers_cv8\\matica_2.txt");
        MatCRS A = load_matrix_crs("C:\\Users\\student\\Harmaniak\\NMLA\\Solvers_cv8\\maticaCG.txt", n1);
        /*MatCRS B = load_matrix_crs("C:\\Users\\student\\Harmaniak\\NMLA\\Solvers_cv8\\matica_2.txt", n2);*/

        Vector b(n1, 1.0);  // prava strana
        Vector x1(n1, 0.0);  // pociatocny odhad BiCG
        Vector x2(n2, 0.0);  // pociatocny odhad BiCGStab
        Vector x3(n1, 0.0); // pociatocny odhad CG
        Vector x4(n1, 0.0); // pociatocny odhad GD

        /*BiCG(A, b, x1, 1000, 1e-8);*/
        BiCGStabCRS(A, b, x1, 1000, 1e-8);
        /*BiCGStabCRS(B, b, x2, 1000, 1e-8);*/
        /*Vector residualsCG = conjugate_gradient(A, b, x3, 1e-8);
        Vector residualsGD = gradient_descent(A, b, x4, 1e-8);*/

        /*cout << "Riesenie maticaCG.txt BiCG:\n";
        for (double xi : x1)
            cout << xi << " ";
        cout << endl;

        cout << "Riesenie maticaCG.txt BiCGStab:\n";
        for (double xi : x2)
            cout << xi << " ";
        cout << endl;

        cout << "Riesenie maticaCG.txt CG:\n";
        for (double xi : x3)
            cout << xi << " ";
        cout << endl;*/
    }
    catch (const exception& e) {
        cerr << "Chyba: " << e.what() << endl;
    }

    return 0;
}
