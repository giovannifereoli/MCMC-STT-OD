// File: two_body_mex.cpp
#include "mex.h"
#include <cmath>

void computeAcceleration(const double* r, double* a, double mu) {
    double rnorm = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    double factor = -mu / (rnorm * rnorm * rnorm);
    a[0] = factor * r[0];
    a[1] = factor * r[1];
    a[2] = factor * r[2];
}

void computeSTMdot(const double* r, const double* v, double* dPhi, const double* Phi, double mu) {
    double rnorm = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

    double I3[9] = {1,0,0, 0,1,0, 0,0,1};
    double dadr[9];
    double rrT[9] = {
        r[0]*r[0], r[0]*r[1], r[0]*r[2],
        r[1]*r[0], r[1]*r[1], r[1]*r[2],
        r[2]*r[0], r[2]*r[1], r[2]*r[2]
    };

    for (int i = 0; i < 9; ++i)
        dadr[i] = -mu * (I3[i]/(rnorm*rnorm*rnorm) - 3 * rrT[i]/pow(rnorm,5));

    // Build A matrix
    double A[36] = {0};
    for (int i = 0; i < 9; ++i) A[i + 18] = I3[i];       // upper right = I
    for (int i = 0; i < 9; ++i) A[i + 27] = dadr[i];     // lower left = dadr

    // Multiply A * Phi
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            dPhi[i*6 + j] = 0;
            for (int k = 0; k < 6; ++k)
                dPhi[i*6 + j] += A[i*6 + k] * Phi[k*6 + j];
        }
    }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    const double* x = mxGetPr(prhs[0]);
    double mu = mxGetScalar(prhs[1]);

    plhs[0] = mxCreateDoubleMatrix(42, 1, mxREAL);
    double* dx = mxGetPr(plhs[0]);

    // r and v
    double r[3] = {x[0], x[1], x[2]};
    double v[3] = {x[3], x[4], x[5]};

    // Compute acceleration
    double a[3];
    computeAcceleration(r, a, mu);

    // Fill derivatives: dx = [v; a; dPhi(:)]
    for (int i = 0; i < 3; ++i) dx[i] = v[i];
    for (int i = 0; i < 3; ++i) dx[3+i] = a[i];

    // STM (6x6)
    const double* Phi = x + 6;
    double* dPhi = dx + 6;
    computeSTMdot(r, v, dPhi, Phi, mu);
}
