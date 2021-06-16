#ifndef SAMPLE_BNP_FIXT0I0_H
#define SAMPLE_BNP_FIXT0I0_H

#include <RcppArmadillo.h>

typedef std::tuple< arma::vec, arma::vec, double, arma::vec, arma::vec, arma::vec > sample_BNP_t;

sample_BNP_t sample_BNP(arma::mat theta, arma::vec psi, arma::cube UY, arma::mat UX, arma::vec eta_theta, arma::vec eta_psi, arma::vec s, arma::vec nj, double K_I, arma::vec rho_star, arma::vec theta0_star, arma::vec psi0_star, double kappa, double a_rho, double b_rho, double m_theta0, double sig2_theta0, double m_psi0, double sig2_psi0, double sig2_eps_theta, double sig2_eps_psi);
  
#endif