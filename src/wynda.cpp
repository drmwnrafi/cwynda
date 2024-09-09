#include <iostream>
#include <Eigen/Dense>
#include "wynda.h"
#include <tuple>

using namespace std;
using namespace Eigen;

class WyNDA {
private:
    MatrixXd R_state;
    MatrixXd R_params;
    MatrixXd P_state;
    MatrixXd P_params;
    MatrixXd Gamma;
    MatrixXd K_state;
    MatrixXd K_params;
    VectorXd state;
    VectorXd params;
    int n_state;
    int n_params;
    const float lambda_state;
    const float lambda_params;

public:
    WyNDA(const int n_state, const int n_params, 
          const VectorXd &init_state = VectorXd(),
          float lambda_state = 0.65,
          float lambda_params = 0.995,
          MatrixXd R_state = MatrixXd(),
          MatrixXd R_params = MatrixXd(),
          MatrixXd P_state = MatrixXd(),
          MatrixXd P_params = MatrixXd()) 
        : n_state(n_state), n_params(n_params), lambda_state(lambda_state), lambda_params(lambda_params){

        if (R_state.size() == 0) {
            this->R_state = 0.1 * MatrixXd::Identity(n_state, n_state);
        } else {
            this->R_state = R_state;
        }

        if (R_params.size() == 0) {
            this->R_params = 10 * MatrixXd::Identity(n_state, n_state);
        } else {
            this->R_params = R_params;
        }

        if (P_state.size() == 0) {
            this->P_state = 10 * MatrixXd::Identity(n_state, n_state);
        } else {
            this->P_state = P_state;
        }

        if (P_params.size() == 0) {
            this->P_params = MatrixXd::Identity(n_params, n_params);
        } else {
            this->P_params = P_params;
        }

        if (init_state.size() == 0) {
            this->state = VectorXd::Zero(n_state);
        } else {
            this->state = init_state;
        }

        this->Gamma = MatrixXd::Identity(n_state, n_params);
        this->K_state = MatrixXd::Zero(n_state, n_state);
        this->K_params = MatrixXd::Zero(n_params, n_state);
        this->params = VectorXd::Zero(n_params);
    }

    void update_gain(){
        this->K_state = this->P_state * (this->P_state + this->R_state).inverse();
        MatrixXd Gamma_T = this->Gamma.transpose();
        this->K_params = this->P_params * Gamma_T * (this->Gamma * this->P_params * Gamma_T + this->R_params).inverse();
        this->Gamma = (MatrixXd::Identity(n_state, n_state) - this->K_state) * this->Gamma;
    }

    void update_model( MatrixXd base, MatrixXd Phi, float dt){
        this->state = this->state + base * dt + Phi * this->params;
        this->params = this->params;
        this->P_state = (1/this->lambda_state)*(MatrixXd::Identity(n_state, n_state) - this->K_state) *  this->P_state;
        this->P_params = (1/this->lambda_params)*(MatrixXd::Identity(n_params, n_params) - this->K_params * this->Gamma)*this->P_params;
        this->Gamma = this->Gamma - Phi;
    }

    void estimate(VectorXd input){
        this->state = this->state + (this->K_state + this->Gamma * this->K_params) * (input - this->state);
        this->params = this->params - this->K_params * (input - this->state);
    }

    pair<VectorXd, VectorXd> run(VectorXd input, MatrixXd wide_array, float dt, VectorXd base = VectorXd()){
        if (base.size() == 0) {
            base = VectorXd::Zero(n_state);
        }
        this->update_gain();
        this->estimate(input);
        this->update_model(base, wide_array, dt);
        return make_pair(this->state, this->params);
    }
};

int main() {
    const int n_state = 6;
    const int n_params = 4;

    VectorXd init_state(n_state);
    init_state << 1, 2, 3, 4, 5, 6;

    WyNDA wynda(n_state, n_params, init_state);

    VectorXd input(n_state);
    input << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5; 

    MatrixXd wide_array(n_state, n_params);
    wide_array = MatrixXd::Random(n_state, n_params);

    float dt = 0.1;

    VectorXd state_estimate, params_estimate;
    tie(state_estimate, params_estimate) = wynda.run(input, wide_array, dt);

    cout << "Estimated State:\n" << state_estimate << endl;
    cout << "Estimated Params:\n" << params_estimate << endl;

    return 0;
}

extern "C" {

    WyNDAHandle create_wynda(int n_state, int n_params, const double* init_state, float lambda_state, float lambda_params) {
        VectorXd init_state_vec = Map<const VectorXd>(init_state, n_state);
        return new WyNDA(n_state, n_params, init_state_vec, lambda_state, lambda_params);
    }

    void destroy_wynda(WyNDAHandle handle) {
        delete static_cast<WyNDA*>(handle);
    }

    void run_wynda(WyNDAHandle handle, const double* input, const double* base, const double* wide_array, int n_state, int n_params, float dt, double* out_state, double* out_params) {
        WyNDA* wynda = static_cast<WyNDA*>(handle);

        VectorXd input_vec = Map<const VectorXd>(input, n_state);
        VectorXd base_vec = Map<const VectorXd>(base, n_state);
        MatrixXd wide_array_mat = Map<const MatrixXd>(wide_array, n_state, n_params);

        VectorXd state_estimate, params_estimate;
        std::tie(state_estimate, params_estimate) = wynda->run(input_vec, wide_array_mat, dt, base_vec);

        Map<VectorXd>(out_state, n_state) = state_estimate;
        Map<VectorXd>(out_params, n_params) = params_estimate;
    }
}
