#ifndef WYND_H
#define WYND_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* WyNDAHandle;

WyNDAHandle create_wynda(int n_state, int n_params, const double* init_state, float lambda_state, float lambda_params);
void destroy_wynda(WyNDAHandle handle);

void run_wynda(WyNDAHandle handle, const double* input, const double* base, const double* wide_array, int n_state, int n_params, float dt, double* out_state, double* out_params);

#ifdef __cplusplus
}
#endif

#endif // WYND_H
