//
// Created by pierfied on 10/3/17.
//

#include <hmc.h>

#ifndef LIKELIHOOD_LIKELIHOOD_H
#define LIKELIHOOD_LIKELIHOOD_H

typedef struct {
    int num_params;
    double *f;
    int *y_inds;
    double *N;
    int nx;
    int ny;
    int nz;
    double mu;
    double *inv_cov;
    double expected_N;
} LikelihoodArgs;

SampleChain sample_map(double *y0, double *m, LikelihoodArgs args,
                       int num_samps, int num_steps, int num_burn,
                       double epsilon);

Hamiltonian map_likelihood(double *y, void *args_ptr);

double log_factorial(int x);

#endif //LIKELIHOOD_LIKELIHOOD_H
