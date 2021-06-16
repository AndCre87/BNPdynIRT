// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <tuple>
#include "sample_BNP.h"
using namespace Rcpp;

sample_BNP_t sample_BNP(arma::mat theta, arma::vec psi, arma::cube UY, arma::mat UX, arma::vec eta_theta, arma::vec eta_psi, arma::vec s, arma::vec nj, double K_I, arma::vec rho_star, arma::vec theta0_star, arma::vec psi0_star, double kappa, double a_rho, double b_rho, double m_theta0, double sig2_theta0, double m_psi0, double sig2_psi0, double sig2_eps_theta, double sig2_eps_psi){
  
  // Extract sizes
  int I = theta.n_rows, TY = theta.n_cols;
  //Number of auxiliary variables for Neal's Alg8, sample from P0
  int N_aux = 50;
  
  double s_rho = 0.25;
  
  // Alg8: Renovate auxiliary variables and initialise vector of uniform probabilities
  arma::vec prob_aux(N_aux, arma::fill::zeros), theta0_aux(N_aux, arma::fill::zeros), psi0_aux(N_aux, arma::fill::zeros), rho_aux(N_aux, arma::fill::zeros);
  for(int i_aux = 0; i_aux < N_aux; i_aux++){
    prob_aux(i_aux) = 1.0 / N_aux;
    theta0_aux(i_aux) = m_theta0 + arma::randn() * sqrt(sig2_theta0);
    psi0_aux(i_aux) = m_psi0 + arma::randn() * sqrt(sig2_psi0);
    rho_aux(i_aux) = - 1 + 2 * R::rbeta(a_rho, b_rho);
  }
  arma::vec prob_aux_cum = arma::cumsum(prob_aux);
  
  
  // Update allocation of i-th subject
  for(int i = 0; i < I; i++){ 
    int aux_s = s(i);
    arma::mat UY_i = UY.row(i);
    arma::rowvec UX_i = UX.row(i);
    
    // Remove element from count
    nj(aux_s) = nj(aux_s) -  1;
    // It could be the only element with j-th label
    bool alone_i = false; 
    if(nj(aux_s) == 0){
      alone_i = true;
    }
    
    // Re-use algorithm
    if(alone_i){
      double aux_runif = arma::randu();
      arma::uvec hh_vec = arma::find(prob_aux_cum >= aux_runif, 1, "first");
      rho_aux(hh_vec(0)) = rho_star(aux_s);
      theta0_aux(hh_vec(0)) = theta0_star(aux_s);
      psi0_aux(hh_vec(0)) = psi0_star(aux_s);
    }
    
    arma::vec f_k(K_I + N_aux, arma::fill::zeros), w(K_I + N_aux, arma::fill::zeros);
    
    // Probability of allocating a new cluster
    for(int i_aux = 0; i_aux < N_aux; i_aux++){
      
      //theta part
      int t = 0;
      arma::rowvec UY_it = UY_i.row(t);
      double m_theta_i_t = theta0_aux(i_aux) + arma::accu(UY_it % eta_theta.t());
      f_k(i_aux) += arma::log_normpdf(theta(i,t), m_theta_i_t, sqrt(sig2_eps_theta));
      for(int t = 1; t < TY; t++){
        arma::rowvec UY_it = UY_i.row(t);
        double m_theta_i_t = theta0_aux(i_aux) + rho_aux(i_aux) * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
        f_k(i_aux) += arma::log_normpdf(theta(i,t), m_theta_i_t, sqrt(sig2_eps_theta));
      }
      
      //psi part
      double m_psi_i = psi0_aux(i_aux) + arma::accu(UX_i.t() % eta_psi);
      f_k(i_aux) += arma::log_normpdf(psi(i), m_psi_i, sqrt(sig2_eps_psi));
    }
    
    // Probability of allocating to an existing cluster
    for(int j = 0; j < K_I; j++){
      
      //theta part
      int t = 0;
      arma::rowvec UY_it = UY_i.row(t);
      double m_theta_it = theta0_star(j) + arma::accu(UY_it % eta_theta.t());
      f_k(N_aux + j) += arma::log_normpdf(theta(i,t), m_theta_it, sqrt(sig2_eps_theta));
      for(int t = 1; t < TY; t++){
        arma::rowvec UY_it = UY_i.row(t);
        double m_theta_it = theta0_star(j) + rho_star(j) * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
        f_k(N_aux + j) += arma::log_normpdf(theta(i,t), m_theta_it, sqrt(sig2_eps_theta));
      }
      
      //psi part
      double m_psi_i = psi0_star(j) + arma::accu(UX_i.t() % eta_psi);
      f_k(N_aux + j) += arma::log_normpdf(psi(i), m_psi_i, sqrt(sig2_eps_psi));
    }
    
    w.head(N_aux).fill(kappa / N_aux);
    w.tail(K_I) = nj;
    
    // Check for empty cluster conditions
    if(alone_i){ // Fix negative weight
      w(N_aux + aux_s) = 0.0;
    }
    
    
    f_k = exp(f_k-max(f_k)) % w;
    f_k = f_k/sum(f_k);
    
    double aux_runif = arma::randu();
    arma::vec f_k_cum = arma::cumsum(f_k);
    arma::uvec hh_vec = arma::find(f_k_cum >= aux_runif, 1, "first");
    int hh = hh_vec(0);
    
    if(hh < N_aux){ // New cluster
      
      // Select unique value from N_aux available
      if(alone_i){ // Same number of clusters
        nj(aux_s) = 1;
        rho_star(aux_s) = rho_aux(hh);
        theta0_star(aux_s) = theta0_aux(hh);
        psi0_star(aux_s) = psi0_aux(hh);
      }else{ // Additional cluster
        nj.insert_rows(K_I,1);
        nj(K_I) = 1;
        rho_star.insert_rows(K_I,1);
        rho_star(K_I) = rho_aux(hh);
        theta0_star.insert_rows(K_I,1);
        theta0_star(K_I) = theta0_aux(hh);
        psi0_star.insert_rows(K_I,1);
        psi0_star(K_I) = psi0_aux(hh);
        s(i) = K_I; // Allocations s have indexing from 0!
        K_I ++;
      }
      //Restore used auxiliary variable
      theta0_aux(hh) = m_theta0 + arma::randn() * sqrt(sig2_theta0);
      psi0_aux(hh) = m_psi0 + arma::randn() * sqrt(sig2_psi0);
      rho_aux(hh) = - 1 + 2 * R::rbeta(a_rho,b_rho);
      
    }else{ // Old cluster
      
      int hk = hh - N_aux;
      nj(hk) ++;
      s(i) = hk;  // Allocations s have indexing from 0!
      if(alone_i){ // Remove empty cluster
        K_I --;
        nj.shed_row(aux_s);
        rho_star.shed_row(aux_s);
        theta0_star.shed_row(aux_s);
        psi0_star.shed_row(aux_s);
        for(int i2 = 0; i2 < I; i2 ++){
          if(s(i2) > aux_s){ // Allocations s have indexing from 0!
            s(i2) = s(i2) - 1;
          }
        }
      }
    }
  }
  
  
  //////////////////////////
  // Update unique values //
  //////////////////////////
  for(int j = 0; j < K_I; j++){
    
    //////////
    //theta0//
    //////////
    double s_theta0_post = 1 / sig2_theta0 + nj(j) * TY / sig2_eps_theta;
    s_theta0_post = 1 / s_theta0_post;
    double m_theta0_post = m_theta0 / sig2_theta0;
    
    for(int i = 0; i < I; i++){
      int aux_s = s(i);
      if(aux_s == j){
        arma::mat UY_i = UY.row(i);
        
        int t = 0;
        arma::rowvec UY_it = UY_i.row(t);
        m_theta0_post += (theta(i,t) - arma::accu(UY_it % eta_theta.t())) / sig2_eps_theta;
        for(int t = 1; t < TY; t++){
          arma::rowvec UY_it = UY_i.row(t);
          m_theta0_post += (theta(i,t) - rho_star(j) * theta(i,t-1) - arma::accu(UY_it % eta_theta.t())) / sig2_eps_theta;
        }
      }
    }
    m_theta0_post = s_theta0_post * m_theta0_post;
    theta0_star(j) = m_theta0_post + arma::randn() * sqrt(s_theta0_post);
    
    
    ////////
    //psi0//
    ////////
    double s_psi0_post = 1 / sig2_psi0 + nj(j) / sig2_eps_psi;
    s_psi0_post = 1 / s_psi0_post;
    double m_psi0_post = m_psi0 / sig2_psi0;
    for(int i = 0; i < I; i++){
      int aux_s = s(i);
      if(aux_s == j){
        arma::rowvec UX_i = UX.row(i);
        m_psi0_post += (psi(i) - arma::accu(UX_i.t() % eta_psi)) / sig2_eps_psi;
      }
    }
    m_psi0_post = s_psi0_post * m_psi0_post;
    psi0_star(j) = m_psi0_post + arma::randn() * sqrt(s_psi0_post);
    
    
    ///////
    //rho//
    ///////
    double rho_new = log((1 + rho_star(j)) / (1 - rho_star(j))) + arma::randn() * sqrt(s_rho);
    rho_new = (1 - exp(-rho_new))/(1 + exp(-rho_new));
    
    // Evaluate log-ratio // Proposal is not symmetric in rho
    double log_ratio_rho = a_rho * (log(1 + rho_new) - log(1 + rho_star(j))) + b_rho * (log(1 - rho_new) - log(1 - rho_star(j)));
    
    for(int i = 0; i < I; i++){
      int aux_s = s(i);
      
      if(aux_s == j){
        arma::mat UY_i = UY.row(i);
        arma::rowvec UX_i = UX.row(i);
        
        // New phi (t > 0)
        for(int t = 1; t < TY; t++){
          arma::rowvec UY_it = UY_i.row(t);
          double m_theta_it = theta0_star(j) + rho_new * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
          log_ratio_rho += - .5 * pow(theta(i,t) - m_theta_it, 2.0) / sig2_eps_theta;
        }
        
        // Current phi
        for(int t = 1; t < TY; t++){
          arma::rowvec UY_it = UY_i.row(t);
          double m_theta_it = theta0_star(j) + rho_star(j) * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
          
          log_ratio_rho += - ( - .5 * pow(theta(i,t) - m_theta_it, 2.0) / sig2_eps_theta );
        }
      }
    }
    
    double accept_rho = 1.0;
    if( arma::is_finite(log_ratio_rho) ){
      if(log_ratio_rho < 0){
        accept_rho = exp(log_ratio_rho);
      }
    }else{
      accept_rho = 0.0;
    }
    
    if( arma::randu() < accept_rho ){
      rho_star(j) = rho_new;
    }
  }
  
  return sample_BNP_t(s, nj, K_I, theta0_star, psi0_star, rho_star);
}
