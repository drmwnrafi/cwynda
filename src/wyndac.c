#include <stdio.h>
#include <stdlib.h> 
#include "wynda.h"

int main() {
    const int n_state = 6;
    const int n_params = 4;

    double* init_state = (double*)malloc(n_state * sizeof(double));
    double* input = (double*)malloc(n_state * sizeof(double));
    double* base = (double*)malloc(n_state * sizeof(double));
    double* wide_array = (double*)malloc(n_state * n_params * sizeof(double));

    if (!init_state || !input || !base || !wide_array) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n_state; ++i) {
        init_state[i] = i + 1;
        input[i] = 0.5;
        base[i] = 0.0;
    }

    for (int i = 0; i < n_state * n_params; ++i) {
        wide_array[i] = (double)rand() / RAND_MAX * 2 - 1;
    }

    float dt = 0.1;

    WyNDAHandle handle = create_wynda(n_state, n_params, init_state, 0.65, 0.995);

    double* out_state = (double*)malloc(n_state * sizeof(double));
    double* out_params = (double*)malloc(n_params * sizeof(double));

    if (!out_state || !out_params) {
        fprintf(stderr, "Memory allocation failed\n");
        free(init_state);
        free(input);
        free(base);
        free(wide_array);
        destroy_wynda(handle);
        return 1;
    }

    run_wynda(handle, input, base, wide_array, n_state, n_params, dt, out_state, out_params);

    printf("Estimated State:\n");
    for (int i = 0; i < n_state; ++i) {
        printf("%lf ", out_state[i]);
    }
    printf("\n");

    printf("Estimated Params:\n");
    for (int i = 0; i < n_params; ++i) {
        printf("%lf ", out_params[i]);
    }
    printf("\n");

    free(init_state);
    free(input);
    free(base);
    free(wide_array);
    free(out_state);
    free(out_params);
    destroy_wynda(handle);

    return 0;
}
