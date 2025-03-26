#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Mat;

Vec vec_add(const Vec& a, const Vec& b, double alpha = 1.0) {
    Vec result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + alpha * b[i];
    }
    
    return result;
}


Vec mat_vec_mult(const Mat& A, const Vec& x) {
    Vec result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[i].size(); j++)
            result[i] += A[i][j] * x[j];
    return result;
}

double dot(const Vec& a, const Vec& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        sum += a[i] * b[i];
    return sum;
}

double norm(const Vec& a) {
    return sqrt(dot(a, a));
}

vector<double> gradient_descent(const Mat& A, const Vec& b, Vec& x, double tol) {
    vector<double> residuals;
    Vec r = vec_add(b, mat_vec_mult(A, x), -1.0);
    int iter = 0;
    const int maxiter = 1000;
    while (norm(r) >= tol && iter <= maxiter) {
        double alpha = dot(r, r) / dot(r, mat_vec_mult(A, r));
        x = vec_add(x, r, alpha);
        r = vec_add(b, mat_vec_mult(A, x), -1.0);
        residuals.push_back(norm(r));
        iter++;
    }

    return residuals;
}

vector<double> conjugate_gradient(const Mat& A, const Vec& b, Vec& x, double tol) {
    vector<double> residuals;
    Vec r = vec_add(b, mat_vec_mult(A, x), -1.0);
    Vec p = r;
    int iter = 0;
    const int maxiter = 1000;
    while (norm(r) >= tol && iter <= maxiter) {
        Vec Ap = mat_vec_mult(A, p);
        double alpha = dot(r, r) / dot(p, Ap);
        x = vec_add(x, p, alpha);
        Vec r_new = vec_add(r, Ap, -alpha);
        double beta = dot(r_new, r_new) / dot(r, r);
        p = vec_add(r_new, p, beta);
        r = r_new;
        residuals.push_back(norm(r));
        iter++;
    }
    return residuals;
}

Mat load_matrix(const string& filepath, int n) {
    Mat A(n, Vec(n));
    ifstream file;
    file.open(filepath);
    if (!file) {
        cerr << "Subor sa nepodarilo otvorit.\n";
        exit(1);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> A[i][j];
    file.close();
    return A;
}

Vec random_vector(int n) {
    Vec v(n);
    for (int i = 0; i < n; i++)
        v[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    return v;
}

int main(void) {

    // Moja matica
    Mat A = {
        {4, 1, 0},
        {1, 3, 0},
        {0, 0, 2}
    };
    Vec b = { 1, 2, 3 };
    Vec x1 = { 0, 0, 0 };
    Vec x2 = { 0, 0, 0 };

    auto result1 = gradient_descent(A, b, x1, 1e-6);
    auto result2 = conjugate_gradient(A, b, x2, 1e-6);

    cout << "\t\t\t\tRezidua" << endl;
    cout << "Iter\tMetoda najvacsieho spadu\tMetoda zdruzenych gradientov\n";
    size_t max_iters = max(result1.size(), result2.size());
    for (size_t i = 0; i < max_iters; i++) {
        double r1 = i < result1.size() ? result1[i] : 0.0;
        double r2 = i < result2.size() ? result2[i] : 0.0;
        cout << i + 1 << "\t" << r1 << "\t\t\t\t" << r2 << "\n";
    }
	

    // CG matica 300x300
    cout << "\n\t\t\t\tCG Matica 300x300" << endl;

    const int N = 300;
    srand(time(NULL));

    Mat M = load_matrix("C:\\Users\\student\\Harmaniak\\NMLA\\Nestacionarne_metody\\maticaCG.txt", N);
    Vec bM = random_vector(N);
    Vec x0(N, 1.0);

    Vec x1M = x0;
    Vec x2M = x0;

    auto result_gd = gradient_descent(M, bM, x1M, 1e-6);
    auto result_cg = conjugate_gradient(M, bM, x2M, 1e-6);

    cout << "Metoda najvacsieho spadu Iteracie: " << result_gd.size() << "\n";
    cout << "Metoda zdruzenych gradientov Iteracie: " << result_cg.size() << "\n";

    cout << "\t\t\t\tRezidua" << endl;
    cout << "Iter\tMetoda najvacsieho spadu\tMetoda zdruzenych gradientov\n";
    size_t max_print = std::min(std::min(result_gd.size(), result_cg.size()), size_t(45));
    for (size_t i = 0; i < max_print; i++) {
        cout << i + 1 << "\t" << result_gd[i] << "\t\t\t\t" << result_cg[i] << "\n";
    }

    // Matica 2
    /*cout << "\n\t\t\t\tCG 2. Matica" << endl;

    const int N = 300;
    srand(time(NULL));

    Mat M2 = load_matrix("C:\\Users\\student\\Harmaniak\\NMLA\\Nestacionarne_metody\\matica_2.txt", N);
    Vec bM2 = random_vector(N);
    Vec x02(N, 1.0);

    Vec x1M2 = x02;
    Vec x2M2 = x02;

    auto result_gd2 = gradient_descent(M2, bM2, x1M2, 1e-6);
    auto result_cg2 = conjugate_gradient(M2, bM2, x2M2, 1e-6);

    cout << "Metoda najvacsieho spadu Iteracie: " << result_gd2.size() << "\n";
    cout << "Metoda zdruzenych gradientov Iteracie: " << result_cg2.size() << "\n";

    cout << "\t\t\t\tRezidua" << endl;
    cout << "Iter\tMetoda najvacsieho spadu\tMetoda zdruzenych gradientov\n";
    size_t max_print2 = std::min(std::min(result_gd2.size(), result_cg2.size()), size_t(50));
    for (size_t i = 0; i < max_print2; i++) {
        cout << i + 1 << "\t" << result_gd2[i] << "\t\t\t\t" << result_cg2[i] << "\n";
    }*/

	return 0;
}
