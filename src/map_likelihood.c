//
// Created by pierfied on 10/3/17.
//

#include "map_likelihood.h"
#include <math.h>
#include <hmc.h>
#include <stdlib.h>
#include <omp.h>

SampleChain sample_map(double *y0, double *m, LikelihoodArgs args,
                       int num_samps, int num_steps, int num_burn,
                       double epsilon) {
    HMCArgs hmc_args;
    hmc_args.log_likelihood = map_likelihood;
    hmc_args.likelihood_args = &args;
    hmc_args.num_samples = num_samps;
    hmc_args.num_params = args.num_params;
    hmc_args.num_steps = num_steps;
    hmc_args.num_burn = num_burn;
    hmc_args.epsilon = epsilon;
    hmc_args.x0 = y0;
    hmc_args.m = m;

    SampleChain chain = hmc(hmc_args);

    return chain;
}

Hamiltonian map_likelihood(double *y, void *args_ptr) {
    LikelihoodArgs *args = (LikelihoodArgs *) args_ptr;

    double mu = args->mu;
    double *inv_cov = args->inv_cov;
    double expected_N = args->expected_N;

    int nx = args->nx;
    int ny = args->ny;
    int nz = args->nz;

    double *f = args->f;
    double *N = args->N;

    int *y_inds = args->y_inds;

    int num_params = args->num_params;
    double *grad = malloc(sizeof(double) * num_params);

    int log_fac_lim = 1000;
    double log_fac[log_fac_lim];
    for (int i = 0; i < log_fac_lim; i++) {
        log_fac[i] = log_factorial(i);
    }

    double normal_contrib = 0;
    double poisson_contrib = 0;
#pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                // Check that this is a high occupancy voxel.
                int ind1 = i * ny * nz + j * nz + k;
                if (y_inds[ind1] < 0) continue;
                int y1 = y_inds[ind1];

                // Loop over neighbors.
                double neighbor_contrib = 0;
                for (int a = 0; a < nx; a++) {
                    for (int b = 0; b < ny; b++) {
                        for (int c = 0; c < nz; ++c) {
                            // Check that this neighbor is high-occupancy.
                            int ind2 = a * ny * nz + b * nz + c;
                            if (y_inds[ind2] < 0) continue;
                            int y2 = y_inds[ind2];

                            // Compute the neighbor contribution.
                            int ic_ind = ind1 * nx * ny * nz + ind2;
                            neighbor_contrib += inv_cov[ic_ind] * (y[y2] - mu);
                        }
                    }
                }

                // Compute the mean of the Poisson term.
                double lambda = expected_N * f[ind1] * exp(y[y1]);

#pragma omp critical
                {
                    // Compute the total contribution of this voxel to the normal.
                    normal_contrib += (y[y1] - mu) * neighbor_contrib;

                    // Compute the Poisson contribution of this voxel.
                    poisson_contrib += N[ind1] * log(lambda) - lambda
                                       - log_fac[lround(N[ind1])];
                }

                // Compute the gradient for the voxel.
                grad[y1] = -neighbor_contrib + N[ind1] - lambda;
            }
        }
    }
    normal_contrib *= -0.5;

    Hamiltonian likelihood;
    likelihood.log_likelihood = normal_contrib + poisson_contrib;
    likelihood.grad = grad;

    return likelihood;
}

double log_factorial(int x) {
    double sum = 0;
    for (int i = 2; i <= x; i++) {
        sum += log(i);
    }
    return sum;
}