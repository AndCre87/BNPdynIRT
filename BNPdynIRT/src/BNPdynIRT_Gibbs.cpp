// Main Gibbs function for PCM BNP IRT model

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <tuple>
#include "sample_BNP.h"
#include <progress.hpp>
#include <progress_bar.hpp>
using namespace Rcpp;

// [[Rcpp::export]]
List BNPdynIRT_Gibbs(List data_list, List MCMC_list, List Param_list){
  
  //Extract data
  List Y = data_list["Y"], Y_bis = clone(Y), X = data_list["X"], X_bis = clone(X), qn_index = data_list["qn_index"];
  arma::cube UY = data_list["UY"];
  arma::mat UX = data_list["UX"];
  
  // Extract sizes
  int TY = Y.size(), QX = X.size();
  arma::mat Y_t = Y[0];
  int I = Y_t.n_rows, qY = UY.n_slices, qX = UX.n_cols;
  
  // Extract number of questions within CDI2 and MASC2
  arma::vec JY(TY, arma::fill::zeros), JX(QX, arma::fill::zeros);
  for(int t = 0; t < TY; t++){
    arma::mat Y_t = Y[t];
    JY(t) = Y_t.n_cols;
  }
  for(int q = 0; q < QX; q++){
    arma::mat X_q = X[q];
    JX(q) = X_q.n_cols;
  }
  // Find max number of unique questions through time points
  int max_JY = 0;
  for(int t = 0; t < TY; t++){
    arma::vec qn_index_t = qn_index[t];
    for(int ind = 0; ind < JY(t); ind++){
      if((qn_index_t(ind) + 1) > max_JY){
        max_JY = qn_index_t(ind) + 1;
      }
    }
  }
  
  //Item parameters and Hyperparameters needed
  bool fix_betas = Param_list["fix_betas"]; // This is to introduce a constraint on the beta coefficients. Sum_h beta_jh (excluding alpha) = 0
  arma::mat Sigma_gamma_Y = as<arma::mat>(Param_list["Sigma_gamma_Y"]), Sigma_gamma_Y_inv = arma::inv_sympd(Sigma_gamma_Y);
  arma::field<arma::mat> Sigma_gamma_X = as<arma::field<arma::mat>>(Param_list["Sigma_gamma_X"]), Sigma_gamma_X_inv(QX);
  arma::vec mu_gamma_Y = as<arma::vec>(Param_list["mu_gamma_Y"]);
  arma::field<arma::mat> gamma_X(QX);
  arma::field<arma::vec> mu_gamma_X = as<arma::field<arma::vec>>(Param_list["mu_gamma_X"]);
  //Number of categories is one less then tem parameters in the vector (exclude alpha!)
  arma::vec mX(QX, arma::fill::zeros), mX_p1(QX, arma::fill::zeros);
  int mY_p1 = mu_gamma_Y.n_elem, mY = 0;
  if(fix_betas){
    mY = mY_p1;
  }else{
    mY = mY_p1 - 1;
  }
  // Indices for constrained proposal
  arma::uvec ind_Y = arma::regspace<arma::uvec>(0, mY);
  ind_Y.shed_row(1);
  arma::field<arma::uvec> ind_X(QX);
  arma::mat gamma_Y(max_JY, mY + 1, arma::fill::zeros);
  for(int q = 0; q < QX; q++){
    arma::vec mu_gamma_X_q = mu_gamma_X[q];
    if(fix_betas){
      mX_p1(q) = mu_gamma_X_q.n_elem;
      mX(q) = mX_p1(q);
    }else{
      mX_p1(q) = mu_gamma_X_q.n_elem;
      mX(q) = mX_p1(q) - 1;
    }
    arma::mat gamma_X_q(JX(q), mX(q) + 1, arma::fill::zeros);
    gamma_X[q] = gamma_X_q;
    arma::mat Sigma_gamma_X_q = Sigma_gamma_X[q];
    Sigma_gamma_X_inv[q] = arma::inv_sympd(Sigma_gamma_X_q);
    // Indices for constrained proposal
    arma::uvec ind_X_q = arma::regspace<arma::uvec>(0, mX(q));
    ind_X_q.shed_row(1);
    ind_X[q] = ind_X_q;
  }
  
  
  // Adaptive MH
  arma::cube S_gamma_Y(mY_p1, mY_p1, max_JY, arma::fill::zeros), prod_gamma_Y(mY_p1, mY_p1, max_JY, arma::fill::zeros);
  arma::field<arma::cube> S_gamma_X(QX), prod_gamma_X(QX);
  arma::field<arma::mat> sum_gamma_X(QX), eye_mat_gamma_X(QX);
  arma::mat sum_gamma_Y(max_JY, mY_p1, arma::fill::zeros), eye_mat_gamma_Y(mY_p1, mY_p1, arma::fill::eye);
  arma::vec s_d_gamma_Y(max_JY, arma::fill::zeros), gamma_Y_accept(max_JY, arma::fill::zeros);
  s_d_gamma_Y.fill(pow(2.4, 2.0) / mY_p1);
  arma::field<arma::vec> s_d_gamma_X(QX), gamma_X_accept(QX);
  for(int j = 0; j < max_JY; j++){
    S_gamma_Y.slice(j).diag().fill(0.01);
  }
  for(int q = 0; q < QX; q++){
    // gamma_X
    arma::cube S_gamma_X_q(mX_p1(q), mX_p1(q), JX(q), arma::fill::zeros), prod_gamma_X_q(mX_p1(q), mX_p1(q), JX(q), arma::fill::zeros);
    arma::mat sum_gamma_X_q(JX(q), mX_p1(q), arma::fill::zeros);
    arma::vec s_d_gamma_X_q(JX(q), arma::fill::zeros), gamma_X_accept_q(JX(q), arma::fill::zeros);
    s_d_gamma_X_q.fill(pow(2.4, 2.0) / mX_p1(q));
    for(int j = 0; j < JX(q); j++){
      S_gamma_X_q.slice(j).diag().fill(0.01);
    }
    S_gamma_X[q] = S_gamma_X_q;
    prod_gamma_X[q] = prod_gamma_X_q;
    sum_gamma_X[q] = sum_gamma_X_q;
    s_d_gamma_X[q] = s_d_gamma_X_q;
    gamma_X_accept[q] = gamma_X_accept_q;
    arma::mat eye_mat_gamma_X_q(mX_p1(q), mX_p1(q), arma::fill::eye);
    eye_mat_gamma_X[q] = eye_mat_gamma_X_q;
  }
  
  
  // Subject parameters
  bool update_sig2_eps_theta = Param_list["update_sig2_eps_theta"], update_sig2_eps_psi = Param_list["update_sig2_eps_psi"];
  double sig2_eps_theta = Param_list["sig2_eps_theta"], sig2_eps_psi = Param_list["sig2_eps_psi"], a_sig2_eps_theta = 0.0, b_sig2_eps_theta = 0.0, a_sig2_eps_psi = 0.0, b_sig2_eps_psi = 0.0;
  if(update_sig2_eps_theta){
    a_sig2_eps_theta = Param_list["a_sig2_eps_theta"];
    b_sig2_eps_theta = Param_list["b_sig2_eps_theta"];
  }
  if(update_sig2_eps_psi){
    a_sig2_eps_psi = Param_list["a_sig2_eps_psi"];
    b_sig2_eps_psi = Param_list["b_sig2_eps_psi"];
  }
  
  // Initialize subject parameter
  arma::mat theta(I, TY, arma::fill::zeros), s_theta(I, TY, arma::fill::ones), theta_accept(I, TY, arma::fill::zeros);
  s_theta.fill(0.01);
  arma::vec psi(I, arma::fill::zeros), s_psi(I, arma::fill::ones), psi_accept(I, arma::fill::zeros);
  s_psi.fill(0.01);
  
  // Regression on subject parameters
  arma::vec m_eta_theta = as<arma::vec>(Param_list["m_eta_theta"]), m_eta_psi = as<arma::vec>(Param_list["m_eta_psi"]), eta_theta = m_eta_theta, eta_psi = m_eta_psi;
  arma::mat V_eta_theta = as<arma::mat>(Param_list["V_eta_theta"]), V_eta_theta_inv = arma::inv_sympd(V_eta_theta), V_eta_psi = as<arma::mat>(Param_list["V_eta_psi"]), V_eta_psi_inv = arma::inv_sympd(V_eta_psi);
  
  
  // Process P
  List process_P = Param_list["process_P"];
  bool update_s = process_P["update_s"], update_kappa = process_P["update_kappa"];
  double kappa = process_P["kappa"], a_kappa = 0.0, b_kappa = 0.0;
  if(update_kappa){
    a_kappa = process_P["a_kappa"];
    b_kappa = process_P["b_kappa"];
  }
  
  //Initialize BNP lists
  arma::vec s = MCMC_list["s_init"], s_aux = unique(s); // allocation variables s
  double K_I = s_aux.n_elem;
  arma::vec nj(K_I,arma::fill::zeros);
  for(int i = 0; i < I; i++){
    nj(s(i)) ++;
  }
  double m_theta0 = Param_list["m_theta0"], sig2_theta0 = Param_list["sig2_theta0"], m_psi0 = Param_list["m_psi0"], sig2_psi0 = Param_list["sig2_psi0"], a_rho = Param_list["a_rho"], b_rho = Param_list["b_rho"];
  arma::vec theta0_star = m_theta0 + arma::randn(K_I, 1) * sqrt(sig2_theta0), psi0_star = m_psi0 + arma::randn(K_I, 1) * sqrt(sig2_psi0), rho_star(K_I, arma::fill::zeros);
  double s_rho = 0.25;
  for(int j = 0; j < K_I; j++){
    rho_star(j) = - 1 + 2 * R::rbeta(a_rho, b_rho);
  }
  
  
  //Algorithm parameters
  double n_burn1 = MCMC_list["n_burn1"], n_burn2 = MCMC_list["n_burn2"], thin = MCMC_list["thin"], n_save = MCMC_list["n_save"];
  int n_tot = n_burn1 + n_burn2 + thin * n_save, iter;
  //Adaptation
  NumericVector ADAPT(4);
  ADAPT(0) = n_burn1; //"burn-in" for adaptation
  ADAPT(1) = 0.7; //exponent for adaptive step
  ADAPT(2) = 0.234; //reference acceptance rate
  ADAPT(3) = 0.001; //for multivariate updates
  
  
  //Output lists
  arma::vec K_I_out(n_save,arma::fill::zeros), sig2_eps_theta_out(n_save,arma::fill::zeros), sig2_eps_psi_out(n_save,arma::fill::zeros), kappa_out(n_save,arma::fill::zeros);
  arma::mat psi_out(n_save, I, arma::fill::zeros), s_out(n_save, I, arma::fill::zeros), eta_theta_out(n_save, qY, arma::fill::zeros), eta_psi_out(n_save, qX, arma::fill::zeros);
  arma::cube theta_out(I,TY,n_save,arma::fill::zeros), gamma_Y_out(max_JY, mY + 1, n_save, arma::fill::zeros);
  List Y_out(n_save), X_out(n_save), rho_star_out(n_save), theta0_star_out(n_save), psi0_star_out(n_save), nj_out(n_save), gamma_X_out(n_save);
  
  Progress progr(n_tot, true);
  
  for(int g = 0; g < n_tot; g ++){
    
    // Rcout << "Missing Y \n";
    //////////////////////////
    // Missing values for Y //
    //////////////////////////
    
    //We do this as first step so we avoid numerical problems
    for(int t = 0; t < TY; t ++){
      
      // Data at time t
      arma::mat Y_t = Y[t];
      arma::mat Y_t_bis = Y_bis[t];
      
      //Question indices at time t
      arma::vec qn_index_t = qn_index[t];
      for(int j = 0; j < JY(t); j ++){
        // Question index at time t
        int ind_j = qn_index_t(j);
        
        // Item parmeters for question j
        arma::rowvec gamma_Y_j = gamma_Y.row(ind_j);
        
        arma::vec prob_h(mY), gamma_cum_h = arma::cumsum(gamma_Y_j.t()) - gamma_Y_j(0);
        for(int i = 0; i < I; i ++){
          
          if(!arma::is_finite(Y_t(i,j))){
            
            prob_h.zeros();
            for(int h = 0; h < mY; h++){
              prob_h(h) = exp(gamma_Y_j(0)) * (h * theta(i,t) - gamma_cum_h(h+1));
            }
            prob_h = exp(prob_h - max(prob_h));
            prob_h = prob_h/sum(prob_h);
            
            double aux_runif = arma::randu();
            arma::vec prob_h_cum = arma::cumsum(prob_h);
            arma::uvec hh_vec = arma::find(prob_h_cum >= aux_runif, 1, "first");
            int hh = hh_vec(0);
            
            Y_t_bis(i,j) = hh;
          }
        }
      }
      Y_bis[t] = Y_t_bis;
    }
    
    
    
    
    // Rcout << "Missing X \n";
    
    //////////////////////////
    // Missing values for X //
    //////////////////////////
    
    for(int q = 0; q < QX; q ++){
      
      // Data for questionnaire q
      arma::mat X_q = X[q];
      arma::mat X_q_bis = X_bis[q];
      
      // Number of categories for questionnaire q
      int mX_q = mX(q);
      
      // Item parmeters for questionnaire q
      arma::mat gamma_X_q = gamma_X[q];
      for(int j = 0; j < JX(q); j ++){
        // Item parmeters for questionnaire j
        arma::rowvec gamma_X_q_j = gamma_X_q.row(j);
        
        arma::vec prob_h(mX_q), gamma_cum_h = arma::cumsum(gamma_X_q_j.t()) - gamma_X_q_j(0);
        for(int i = 0; i < I; i++){
          
          if(!arma::is_finite(X_q(i,j))){
            
            prob_h.zeros();
            for(int h = 0; h < mX_q; h++){
              prob_h(h) = exp(gamma_X_q_j(0)) * (h * psi(i) - gamma_cum_h(h+1));
            }
            prob_h = exp(prob_h - max(prob_h));
            prob_h = prob_h/sum(prob_h);
            
            double aux_runif = arma::randu();
            arma::vec prob_h_cum = arma::cumsum(prob_h);
            arma::uvec hh_vec = arma::find(prob_h_cum >= aux_runif, 1, "first");
            int hh = hh_vec(0);
            
            X_q_bis(i,j) = hh;
          }
        }
      }
      X_bis[q] = X_q_bis;
    }
    
    
    
    if(update_s){
      // Rcout << "BNP \n";
      /////////////////////
      // sample BNP part //
      /////////////////////
      
      std::tie(s, nj, K_I, theta0_star, psi0_star, rho_star) = sample_BNP(theta, psi, UY, UX, eta_theta, eta_psi, s, nj, K_I, rho_star, theta0_star, psi0_star, kappa, a_rho, b_rho, m_theta0, sig2_theta0, m_psi0, sig2_psi0, sig2_eps_theta, sig2_eps_psi);
    }else{
      // Rcout << "Unique values \n";
      //////////////////////////
      // Update unique values //
      //////////////////////////
      
      for(int j = 0; j < K_I; j++){
        
        //////////
        //theta0//
        //////////
        double s_theta0_post = 1 / sig2_theta0 * nj(j) * TY / sig2_eps_theta;
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
            
            // New rho (t > 0)
            for(int t = 1; t < TY; t++){
              arma::rowvec UY_it = UY_i.row(t);
              double m_theta_it = theta0_star(j) + rho_new * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
              log_ratio_rho += - .5 * pow(theta(i,t) - m_theta_it, 2.0) / sig2_eps_theta;
            }
            
            // Current rho
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
    }
    
    
    
    
    
    // Rcout << "theta \n";
    ////////////////////
    // sample theta's //
    ////////////////////
    
    for(int i = 0; i < I; i++){
      int aux_s = s(i);
      arma::mat UY_i = UY.row(i);
      
      // t = 0
      int t = 0;
      arma::rowvec UY_it = UY_i.row(t);
      // Propose a new value
      double theta_new = theta(i,t) + arma::randn() * sqrt(s_theta(i,t));
      
      // Prior part (time t)
      double m_theta_it = theta0_star(aux_s) + arma::accu(UY_it % eta_theta.t());
      double log_ratio_theta = - .5 * (pow(theta_new - m_theta_it, 2) - pow(theta(i,t) - m_theta_it, 2)) / sig2_eps_theta;
      
      // Prior part (time t+1)
      arma::rowvec UY_itplus1 = UY_i.row(t+1);
      m_theta_it = theta0_star(aux_s) + rho_star(aux_s) * theta_new + arma::accu(UY_itplus1 % eta_theta.t());
      log_ratio_theta += - .5 * pow(theta(i,t+1) - m_theta_it, 2) / sig2_eps_theta;
      m_theta_it = theta0_star(aux_s) + rho_star(aux_s) * theta(i,t) + arma::accu(UY_itplus1 % eta_theta.t());
      log_ratio_theta += - ( - .5 * pow(theta(i,t+1) - m_theta_it, 2) / sig2_eps_theta );
      
      // Likelihood part
      arma::mat Y_0_bis = Y_bis[t];
      //Question indices at time t
      arma::vec qn_index_t = qn_index[t];
      arma::vec prob_i0j(mY);
      for(int j = 0; j < JY(t); j++){
        // Question index at time t
        int ind_j = qn_index_t(j);
        
        // Item parmeters for question j
        arma::rowvec gamma_Y_j = gamma_Y.row(ind_j);
        arma::vec gamma_cum_h = arma::cumsum(gamma_Y_j.t()) - gamma_Y_j(0);
        
        prob_i0j.zeros();
        for(int h = 0; h < mY; h++){
          prob_i0j(h) = exp(gamma_Y_j(0)) * (h * theta_new - gamma_cum_h(h+1));
        }
        prob_i0j = exp(prob_i0j - max(prob_i0j));
        prob_i0j = prob_i0j/sum(prob_i0j);
        
        log_ratio_theta += log(prob_i0j(Y_0_bis(i,j)));
        
        prob_i0j.zeros();
        for(int h = 0; h < mY; h++){
          prob_i0j(h) = exp(gamma_Y_j(0)) * (h * theta(i,t) - gamma_cum_h(h+1));
        }
        prob_i0j = exp(prob_i0j - max(prob_i0j));
        prob_i0j = prob_i0j/sum(prob_i0j);
        
        log_ratio_theta += - log(prob_i0j(Y_0_bis(i,j)));
      }
      
      double accept_theta = 1.0;
      if( arma::is_finite(log_ratio_theta) ){
        if(log_ratio_theta < 0){
          accept_theta = exp(log_ratio_theta);
        }
      }else{
        accept_theta = 0.0;
      }
      
      theta_accept(i,t) += accept_theta;
      if( arma::randu() < accept_theta ){
        theta(i,t) = theta_new;
      }
      
      s_theta(i,t) = s_theta(i,t) + pow(g+1,-ADAPT(1)) * (accept_theta - ADAPT(2));
      if(s_theta(i,t) > exp(50)){
        s_theta(i,t) = exp(50);
      }else{
        if(s_theta(i,t) < exp(-50)){
          s_theta(i,t) = exp(-50);
        }
      }
      
      
      // t \in [2,TY-1]
      for(int t = 1; t < (TY-1); t++){
        arma::rowvec UY_it = UY_i.row(t);
        // Propose a new value
        double theta_new = theta(i,t) + arma::randn() * sqrt(s_theta(i,t));
        
        // Prior part (time t)
        double m_theta_it = theta0_star(aux_s) + rho_star(aux_s) * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
        double log_ratio_theta = - .5 * (pow(theta_new - m_theta_it, 2) - pow(theta(i,t) - m_theta_it, 2)) / sig2_eps_theta;
        
        // Prior part (time t+1)
        arma::rowvec UY_itplus1 = UY_i.row(t+1);
        m_theta_it = theta0_star(aux_s) + rho_star(aux_s) * theta_new + arma::accu(UY_itplus1 % eta_theta.t());
        log_ratio_theta += - .5 * pow(theta(i,t+1) - m_theta_it, 2) / sig2_eps_theta;
        m_theta_it = theta0_star(aux_s) + rho_star(aux_s) * theta(i,t) + arma::accu(UY_itplus1 % eta_theta.t());
        log_ratio_theta += - ( - .5 * pow(theta(i,t+1) - m_theta_it, 2) / sig2_eps_theta );
        
        // Likelihood part
        arma::mat Y_t_bis = Y_bis[t];
        //Question indices at time t
        arma::vec qn_index_t = qn_index[t];
        arma::vec prob_itj(mY);
        for(int j = 0; j < JY(t); j++){
          // Question index at time t
          int ind_j = qn_index_t(j);
          
          // Item parmeters for question j
          arma::rowvec gamma_Y_j = gamma_Y.row(ind_j);
          arma::vec gamma_cum_h = arma::cumsum(gamma_Y_j.t()) - gamma_Y_j(0);
          
          prob_itj.zeros();
          for(int h = 0; h < mY; h++){
            prob_itj(h) = exp(gamma_Y_j(0)) * (h * theta_new - gamma_cum_h(h+1));
          }
          prob_itj = exp(prob_itj - max(prob_itj));
          prob_itj = prob_itj/sum(prob_itj);
          
          log_ratio_theta += log(prob_itj(Y_t_bis(i,j)));
          
          prob_itj.zeros();
          for(int h = 0; h < mY; h++){
            prob_itj(h) = exp(gamma_Y_j(0)) * (h * theta(i,t) - gamma_cum_h(h+1));
          }
          prob_itj = exp(prob_itj - max(prob_itj));
          prob_itj = prob_itj/sum(prob_itj);
          
          log_ratio_theta += - log(prob_itj(Y_t_bis(i,j)));
        }
        
        double accept_theta = 1.0;
        if( arma::is_finite(log_ratio_theta) ){
          if(log_ratio_theta < 0){
            accept_theta = exp(log_ratio_theta);
          }
        }else{
          accept_theta = 0.0;
        }
        
        theta_accept(i,t) += accept_theta;
        if( arma::randu() < accept_theta ){
          theta(i,t) = theta_new;
        }
        
        s_theta(i,t) = s_theta(i,t) + pow(g+1,-ADAPT(1)) * (accept_theta - ADAPT(2));
        if(s_theta(i,t) > exp(50)){
          s_theta(i,t) = exp(50);
        }else{
          if(s_theta(i,t) < exp(-50)){
            s_theta(i,t) = exp(-50);
          }
        }
      }
      
      
      
      
      // t = TY
      t = TY-1;
      UY_it = UY_i.row(t);
      // Propose a new value
      theta_new = theta(i,t) + arma::randn() * sqrt(s_theta(i,t));
      
      // Prior part
      m_theta_it = theta0_star(aux_s) + rho_star(aux_s) * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
      log_ratio_theta = - .5 * (pow(theta_new - m_theta_it, 2) - pow(theta(i,t) - m_theta_it, 2)) / sig2_eps_theta;
      
      // Likelihood part
      arma::mat Y_T_bis = Y_bis[t];
      //Question indices at time t
      arma::vec qn_index_T = qn_index[t];
      arma::vec prob_iTj(mY);
      for(int j = 0; j < JY(t); j++){
        // Question index at time t
        int ind_j = qn_index_T(j);
        
        // Item parmeters for question j
        arma::rowvec gamma_Y_j = gamma_Y.row(ind_j);
        arma::vec gamma_cum_h = arma::cumsum(gamma_Y_j.t()) - gamma_Y_j(0);
        
        prob_iTj.zeros();
        for(int h = 0; h < mY; h++){
          prob_iTj(h) = exp(gamma_Y_j(0)) * (h * theta_new - gamma_cum_h(h+1));
        }
        prob_iTj = exp(prob_iTj - max(prob_iTj));
        prob_iTj = prob_iTj/sum(prob_iTj);
        
        log_ratio_theta += log(prob_iTj(Y_T_bis(i,j)));
        
        prob_iTj.zeros();
        for(int h = 0; h < mY; h++){
          prob_iTj(h) = exp(gamma_Y_j(0)) * (h * theta(i,t) - gamma_cum_h(h+1));
        }
        prob_iTj = exp(prob_iTj - max(prob_iTj));
        prob_iTj = prob_iTj/sum(prob_iTj);
        
        log_ratio_theta += - log(prob_iTj(Y_T_bis(i,j)));
      }
      
      accept_theta = 1.0;
      if( arma::is_finite(log_ratio_theta) ){
        if(log_ratio_theta < 0){
          accept_theta = exp(log_ratio_theta);
        }
      }else{
        accept_theta = 0.0;
      }
      
      theta_accept(i,t) += accept_theta;
      if( arma::randu() < accept_theta ){
        theta(i,t) = theta_new;
      }
      
      s_theta(i,t) = s_theta(i,t) + pow(g+1,-ADAPT(1)) * (accept_theta - ADAPT(2));
      if(s_theta(i,t) > exp(50)){
        s_theta(i,t) = exp(50);
      }else{
        if(s_theta(i,t) < exp(-50)){
          s_theta(i,t) = exp(-50);
        }
      }
    }
    
    
    
    
    // Rcout << "psi \n";
    //////////////////
    // sample psi's //
    //////////////////
    
    //Skip first subject for constraints (set to zero)
    for(int i = 0; i < I; i ++){
      int aux_s = s(i);
      arma::rowvec UX_i = UX.row(i);
      
      // Propose a new value
      double psi_new = psi(i) + arma::randn() * sqrt(s_psi(i));
      
      // Prior part
      double m_psi_i = psi0_star(aux_s) + arma::accu(UX_i.t() % eta_psi);
      double log_ratio_psi = - .5 * (pow(psi_new - m_psi_i, 2) - pow(psi(i) - m_psi_i, 2)) / sig2_eps_psi;
      
      for(int q = 0; q < QX; q ++){
        
        // Data for questionnaire q
        arma::mat X_q_bis = X_bis[q];
        // Number of categories for questionnaire q
        int mX_q = mX(q);
        // Item parmeters for questionnaire q
        arma::mat gamma_X_q = gamma_X[q];
        arma::vec prob_ij(mX_q);
        for(int j = 0; j < JX(q); j ++){
          // Item parmeters for questionnaire j
          arma::rowvec gamma_X_q_j = gamma_X_q.row(j);
          arma::vec gamma_cum_h = arma::cumsum(gamma_X_q_j.t()) - gamma_X_q_j(0);
          
          prob_ij.zeros();
          for(int h = 0; h < mX_q; h++){
            prob_ij(h) = exp(gamma_X_q_j(0)) * (h * psi_new - gamma_cum_h(h+1));
          }
          prob_ij = exp(prob_ij - max(prob_ij));
          prob_ij = prob_ij/sum(prob_ij);
          log_ratio_psi += log(prob_ij(X_q_bis(i,j)));
          
          prob_ij.zeros();
          for(int h = 0; h < mX_q; h++){
            prob_ij(h) = exp(gamma_X_q_j(0)) * (h * psi(i) - gamma_cum_h(h+1));
          }
          prob_ij = exp(prob_ij - max(prob_ij));
          prob_ij = prob_ij/sum(prob_ij);
          log_ratio_psi += - log(prob_ij(X_q_bis(i,j)));
        }
      }
      
      double accept_psi = 1.0;
      if( arma::is_finite(log_ratio_psi) ){
        if(log_ratio_psi < 0){
          accept_psi = exp(log_ratio_psi);
        }
      }else{
        accept_psi = 0.0;
      }
      
      psi_accept(i) = psi_accept(i) + accept_psi;
      if( R::runif(0,1) < accept_psi ){
        psi(i) = psi_new;
      }
      
      s_psi(i) = s_psi(i) + pow(g+1,-ADAPT(1)) * (accept_psi - ADAPT(2));
      if(s_psi(i) > exp(50)){
        s_psi(i) = exp(50);
      }else{
        if(s_psi(i) < exp(-50)){
          s_psi(i) = exp(-50);
        }
      }
    }
    
    
    
    
    
    
    
    // Rcout << "gamma_Y \n";
    ////////////////////
    // sample gamma_Y //
    ////////////////////
    
    // For each question, update the corresponding item parameters
    for(int j = 0; j < max_JY; j++){
      
      // Current value of item parameters
      arma::vec gamma_Y_j = gamma_Y.row(j).t();
      // Propose new value of item parameters
      arma::mat S_gamma_Y_j = S_gamma_Y.slice(j);
      arma::vec gamma_Y_j_new = gamma_Y_j;
      if(fix_betas){
        gamma_Y_j_new.elem(ind_Y) = arma::mvnrnd(gamma_Y_j.elem(ind_Y), S_gamma_Y_j);
        gamma_Y_j_new(1) = - (arma::sum(gamma_Y_j_new.elem(ind_Y)) - gamma_Y_j_new(0));
      }else{
        gamma_Y_j_new = arma::mvnrnd(gamma_Y_j, S_gamma_Y_j);
      }
      
      //Computing MH ratio:
      double log_ratio_gamma_Y = 0.0;
      //Prior and proposal (symmetric)
      if(fix_betas){
        log_ratio_gamma_Y += - .5 * (arma::as_scalar((gamma_Y_j_new.elem(ind_Y).t() - mu_gamma_Y.t()) * Sigma_gamma_Y_inv * (gamma_Y_j_new.elem(ind_Y) - mu_gamma_Y) - arma::as_scalar((gamma_Y_j.elem(ind_Y).t() - mu_gamma_Y.t()) * Sigma_gamma_Y_inv * (gamma_Y_j.elem(ind_Y) - mu_gamma_Y))));
      }else{
        log_ratio_gamma_Y += - .5 * (arma::as_scalar((gamma_Y_j_new.t() - mu_gamma_Y.t()) * Sigma_gamma_Y_inv * (gamma_Y_j_new - mu_gamma_Y) - arma::as_scalar((gamma_Y_j.t() - mu_gamma_Y.t()) * Sigma_gamma_Y_inv * (gamma_Y_j - mu_gamma_Y))));
      }
      
      // Consider contribution from all time points
      for(int t = 0; t < TY; t ++){
        
        //Question indices at time t
        arma::vec qn_index_t = qn_index[t];
        arma::uvec ind_j = arma::find(qn_index_t == j, 1, "first");
        if(ind_j.n_elem > 0){// Question exists at this time point?
          
          //Data at time t
          arma::mat Y_t_bis = Y_bis[t];
          arma::vec Y_tj = Y_t_bis.cols(ind_j);
          
          //Likelihood
          arma::vec prob_itj(mY), gamma_cum_h = arma::cumsum(gamma_Y_j) - gamma_Y_j(0), gamma_new_cum_h = arma::cumsum(gamma_Y_j_new) - gamma_Y_j_new(0);
          for(int i = 0; i < I; i ++){
            
            prob_itj.zeros();
            for(int h = 0; h < mY; h++){
              prob_itj(h) = exp(gamma_Y_j_new(0)) * (h * theta(i,t) - gamma_new_cum_h(h+1));
            }
            prob_itj = exp(prob_itj - max(prob_itj));
            prob_itj = prob_itj/sum(prob_itj);
            
            log_ratio_gamma_Y += log(prob_itj(Y_tj(i)));
            
            
            prob_itj.zeros();
            for(int h = 0; h < mY; h++){
              prob_itj(h) = exp(gamma_Y_j(0)) * (h * theta(i,t) - gamma_cum_h(h+1));
            }
            prob_itj = exp(prob_itj - max(prob_itj));
            prob_itj = prob_itj/sum(prob_itj);
            
            log_ratio_gamma_Y += - log(prob_itj(Y_tj(i)));
          }
        }
      }
      
      double accept_gamma_Y = 1.0;
      if( arma::is_finite(log_ratio_gamma_Y) ){
        if(log_ratio_gamma_Y < 0){
          accept_gamma_Y = exp(log_ratio_gamma_Y);
        }
      }else{
        accept_gamma_Y = 0.0;
      }
      
      gamma_Y_accept(j) += accept_gamma_Y;
      
      if( arma::randu() < accept_gamma_Y ){
        // Item parameters at time t for question j
        gamma_Y.row(j) = gamma_Y_j_new.t();
      }
      
      if(fix_betas){
        arma::vec gamma_Y_j_aux = gamma_Y.row(j).t();
        sum_gamma_Y.row(j) += gamma_Y_j_aux.elem(ind_Y).t();
        prod_gamma_Y.slice(j) += gamma_Y_j_aux.elem(ind_Y) * gamma_Y_j_aux.elem(ind_Y).t();
      }else{
        sum_gamma_Y.row(j) += gamma_Y.row(j);
        prod_gamma_Y.slice(j) += gamma_Y.row(j).t() * gamma_Y.row(j);
      }
      
      s_d_gamma_Y(j) += pow(g+1,-ADAPT(1))*(accept_gamma_Y - ADAPT(2));
      if(s_d_gamma_Y(j) > exp(50)){
        s_d_gamma_Y(j) = exp(50);
      }else{
        if(s_d_gamma_Y(j) < exp(-50)){
          s_d_gamma_Y(j) = exp(-50);
        }
      }
      if(g > (ADAPT(0) - 1)){
        S_gamma_Y.slice(j) = s_d_gamma_Y(j)/g * (prod_gamma_Y.slice(j) - sum_gamma_Y.row(j).t() * sum_gamma_Y.row(j)/(g+1.0)) + s_d_gamma_Y(j) * pow(0.1,2.0) / mY_p1 * eye_mat_gamma_Y;
      }
    }
    
    
    
    
    
    
    
    // Rcout << "gamma_X \n";
    ////////////////////
    // sample gamma_X //
    ////////////////////
    
    // Update the item parameters for each questionnaire
    for(int q = 0; q < QX; q ++){
      
      // Number of categories in questionnaire q
      int mx_q = mX(q);
      
      //Data for questionnaire q
      arma::mat X_q_bis = X_bis[q];
      
      // Item (hyper)parameters for questionnaire q
      arma::mat gamma_X_q = gamma_X[q];
      arma::vec mu_gamma_X_q = mu_gamma_X[q];
      arma::mat Sigma_gamma_X_q_inv = Sigma_gamma_X_inv[q];
      
      // Adaptive MH for questionnaire q
      arma::cube S_gamma_X_q = S_gamma_X[q];
      arma::vec gamma_X_accept_q = gamma_X_accept[q];
      arma::mat sum_gamma_X_q = sum_gamma_X[q];
      arma::cube prod_gamma_X_q = prod_gamma_X[q];
      arma::vec s_d_gamma_X_q = s_d_gamma_X[q];
      arma::mat eye_mat_gamma_X_q = eye_mat_gamma_X[q];
      arma::uvec ind_X_q = ind_X[q];
      for(int j = 0; j < JX(q); j++){
        // Current value of item parameters
        arma::vec gamma_X_q_j = gamma_X_q.row(j).t();
        // Propose new value of item parameters
        arma::mat S_gamma_X_q_j = S_gamma_X_q.slice(j);
        arma::vec gamma_X_q_j_new = gamma_X_q_j;
        if(fix_betas){
          gamma_X_q_j_new.elem(ind_X_q) = arma::mvnrnd(gamma_X_q_j.elem(ind_X_q), S_gamma_X_q_j);
          gamma_X_q_j_new(1) = - (arma::sum(gamma_X_q_j_new.elem(ind_X_q)) - gamma_X_q_j_new(0));
        }else{
          gamma_X_q_j_new = arma::mvnrnd(gamma_X_q_j, S_gamma_X_q_j);
        }
        
        //Computing MH ratio:
        double log_ratio_gamma_X = 0.0;
        //Prior and proposal (symmetric)
        if(fix_betas){
          log_ratio_gamma_X += - .5 * (arma::as_scalar((gamma_X_q_j_new.elem(ind_X_q).t() - mu_gamma_X_q.t()) * Sigma_gamma_X_q_inv * (gamma_X_q_j_new.elem(ind_X_q) - mu_gamma_X_q) - arma::as_scalar((gamma_X_q_j.elem(ind_X_q).t() - mu_gamma_X_q.t()) * Sigma_gamma_X_q_inv * (gamma_X_q_j.elem(ind_X_q) - mu_gamma_X_q))));
        }else{
          log_ratio_gamma_X += - .5 * (arma::as_scalar((gamma_X_q_j_new.t() - mu_gamma_X_q.t()) * Sigma_gamma_X_q_inv * (gamma_X_q_j_new - mu_gamma_X_q) - arma::as_scalar((gamma_X_q_j.t() - mu_gamma_X_q.t()) * Sigma_gamma_X_q_inv * (gamma_X_q_j - mu_gamma_X_q))));
        }
        
        
        //Likelihood
        arma::vec prob_iqj(mx_q), gamma_cum_h = arma::cumsum(gamma_X_q_j) - gamma_X_q_j(0), gamma_new_cum_h = arma::cumsum(gamma_X_q_j_new) - gamma_X_q_j_new(0);
        for(int i = 0; i < I; i ++){
          prob_iqj.zeros();
          for(int h = 0; h < mx_q; h++){
            prob_iqj(h) = exp(gamma_X_q_j_new(0)) * (h * psi(i) - gamma_new_cum_h(h+1));
          }
          prob_iqj = exp(prob_iqj - max(prob_iqj));
          prob_iqj = prob_iqj/sum(prob_iqj);
          
          log_ratio_gamma_X += log(prob_iqj(X_q_bis(i,j)));
          
          
          prob_iqj.zeros();
          for(int h = 0; h < mx_q; h++){
            prob_iqj(h) = exp(gamma_X_q_j(0)) * (h * psi(i) - gamma_cum_h(h+1));
          }
          prob_iqj = exp(prob_iqj - max(prob_iqj));
          prob_iqj = prob_iqj/sum(prob_iqj);
          
          log_ratio_gamma_X += - log(prob_iqj(X_q_bis(i,j)));
        }
        
        double accept_gamma_X = 1.0;
        if( arma::is_finite(log_ratio_gamma_X) ){
          if(log_ratio_gamma_X < 0){
            accept_gamma_X = exp(log_ratio_gamma_X);
          }
        }else{
          accept_gamma_X = 0.0;
        }
        
        gamma_X_accept_q(j) += accept_gamma_X;
        
        if( arma::randu() < accept_gamma_X ){
          // Item parameters at time t for question j
          gamma_X_q.row(j) = gamma_X_q_j_new.t();
        }
        
        if(fix_betas){
          arma::vec gamma_X_q_j_aux = gamma_X_q.row(j).t();
          sum_gamma_X_q.row(j) += gamma_X_q_j_aux.elem(ind_X_q).t();
          prod_gamma_X_q.slice(j) += gamma_X_q_j_aux.elem(ind_X_q) * gamma_X_q_j_aux.elem(ind_X_q).t();
        }else{
          sum_gamma_X_q.row(j) += gamma_X_q.row(j);
          prod_gamma_X_q.slice(j) += gamma_X_q.row(j).t() * gamma_X_q.row(j);
        }
        
        s_d_gamma_X_q(j) += pow(g+1,-ADAPT(1))*(accept_gamma_X - ADAPT(2));
        if(s_d_gamma_X_q(j) > exp(50)){
          s_d_gamma_X_q(j) = exp(50);
        }else{
          if(s_d_gamma_X_q(j) < exp(-50)){
            s_d_gamma_X_q(j) = exp(-50);
          }
        }
        if(g > (ADAPT(0) - 1)){
          S_gamma_X_q.slice(j) = s_d_gamma_X_q(j)/g * (prod_gamma_X_q.slice(j) - sum_gamma_X_q.row(j).t() * sum_gamma_X_q.row(j)/(g+1.0)) + s_d_gamma_X_q(j) * pow(0.1,2.0) / mX_p1(q) * eye_mat_gamma_X_q;
        }
      }
      gamma_X[q] = gamma_X_q;
      gamma_X_accept[q] = gamma_X_accept_q;
      sum_gamma_X[q] = sum_gamma_X_q;
      prod_gamma_X[q] = prod_gamma_X_q;
      s_d_gamma_X[q] = s_d_gamma_X_q;
      S_gamma_X[q] = S_gamma_X_q;
    }
    
    
    
    
    
    
    // Rcout << "eta_theta \n";
    //////////////////////
    // sample eta_theta // joint proposal over time points
    //////////////////////
    
    arma::mat Spost_eta_theta = V_eta_theta_inv;
    arma::mat mpost_eta_theta = V_eta_theta_inv * m_eta_theta;
    for(int i = 0; i < I; i ++){
      //Cluster assignment of subject i
      int aux_s = s(i);
      
      //Covariates for subject i
      arma::mat UY_i = UY.row(i);
      
      int t = 0;
      arma::rowvec UY_it = UY_i.row(t);
      Spost_eta_theta += UY_it.t() * UY_it / sig2_eps_theta;
      mpost_eta_theta += UY_it.t() * (theta(i,t) - theta0_star(aux_s)) / sig2_eps_theta;
      for(int t = 1; t < TY; t ++){
        arma::rowvec UY_it = UY_i.row(t);
        Spost_eta_theta += UY_it.t() * UY_it / sig2_eps_theta;
        mpost_eta_theta += UY_it.t() * (theta(i,t) - theta0_star(aux_s) - rho_star(aux_s) * theta(i,t-1)) / sig2_eps_theta;
      }
    }
    Spost_eta_theta = arma::inv_sympd(Spost_eta_theta);
    mpost_eta_theta = Spost_eta_theta * mpost_eta_theta;
    
    eta_theta = arma::mvnrnd(mpost_eta_theta, Spost_eta_theta);
    
    
    
    
    
    // Rcout << "eta_psi \n";
    ////////////////////
    // sample eta_psi // joint proposal over questionnaires
    ////////////////////
    
    arma::mat Spost_eta_psi = V_eta_psi_inv;
    arma::vec mpost_eta_psi = V_eta_psi_inv * m_eta_psi;
    for(int i = 0; i < I; i ++){
      //Cluster assignment of subject i
      int aux_s = s(i);
      
      //Covariates for subject i
      arma::rowvec UX_i = UX.row(i);
      Spost_eta_psi += UX_i.t() * UX_i / sig2_eps_psi;
      mpost_eta_psi += UX_i.t() * (psi(i) - psi0_star(aux_s)) / sig2_eps_psi;
      
    }
    Spost_eta_psi = arma::inv_sympd(Spost_eta_psi);
    mpost_eta_psi = Spost_eta_psi * mpost_eta_psi;
    
    eta_psi = arma::mvnrnd(mpost_eta_psi, Spost_eta_psi);
    
    
    
    
    
    if(update_sig2_eps_theta){
      // Rcout << "sig2_eps_theta \n";
      ///////////////////////////
      // sample sig2_eps_theta //
      ///////////////////////////
      
      // Conjugate inverse-gamma prior
      double aux_sig2_eps_theta = b_sig2_eps_theta;
      for(int i = 0; i < I; i++){
        int aux_s = s(i);
        arma::mat UY_i = UY.row(i);
        
        int t = 0;
        arma::rowvec UY_it = UY_i.row(t);
        double m_theta_it = theta0_star(aux_s) + arma::accu(UY_it % eta_theta.t());
        aux_sig2_eps_theta += 0.5 * pow(theta(i,t) - m_theta_it, 2);
        for(int t = 1; t < TY; t++){
          arma::rowvec UY_it = UY_i.row(t);
          double m_theta_it = theta0_star(aux_s) + rho_star(aux_s) * theta(i,t-1) + arma::accu(UY_it % eta_theta.t());
          aux_sig2_eps_theta += 0.5 * pow(theta(i,t) - m_theta_it, 2);
        }
      }
      sig2_eps_theta = 1/arma::randg(arma::distr_param(a_sig2_eps_theta + I * TY / 2, 1/aux_sig2_eps_theta ));
    }
    
    
    
    if(update_sig2_eps_psi){
      // Rcout << "sig2_eps_psi \n";
      /////////////////////////
      // sample sig2_eps_psi //
      /////////////////////////
      
      // Conjugate inverse-gamma prior
      double aux_sig2_eps_psi = b_sig2_eps_psi;
      for(int i = 1; i < I; i++){
        int aux_s = s(i);
        
        arma::rowvec UX_i = UX.row(i);
        double m_psi_i = psi0_star(aux_s) + arma::accu(UX_i.t() % eta_psi);
        aux_sig2_eps_psi += 0.5 * pow(psi(i) - m_psi_i, 2);
      }
      sig2_eps_psi = 1/arma::randg(arma::distr_param(a_sig2_eps_psi + I / 2, 1/aux_sig2_eps_psi ));
    }
    
    
    
    
    if(update_kappa){
      // Rcout << "kappa \n";
      //////////////////
      // sample kappa //
      //////////////////
      
      double kappa_aux = R::rbeta(kappa + 1, I);
      
      double pi_kappa1 = a_kappa + K_I - 1;
      double pi_kappa2 = b_kappa - log(kappa_aux);
      double pi_kappa = pi_kappa1 / (pi_kappa1 + I * pi_kappa2 );
      
      if( arma::randu() < pi_kappa){
        kappa = arma::randg(arma::distr_param(pi_kappa1 + 1, 1/pi_kappa2));
      }else{
        kappa = arma::randg(arma::distr_param(pi_kappa1, 1/pi_kappa2));
      }
    }
    
    
    
    if( (g + 1) % 100 == 0 ){
      Rcout << "g = " << g + 1 << "\n";
      
      Rcout << "K_I = " << K_I << "\n";
      Rcout << "nj = " << nj.t() << "\n";
      
      Rcout << "min(theta) = " << min(theta) << "\n";
      Rcout << "max(theta) = " << max(theta) << "\n";
      
      Rcout << "min(psi) = " << min(psi) << "\n";
      Rcout << "max(psi) = " << max(psi) << "\n";
    }
    
    
    
    
    //Save output for this iteration
    if( (g + 1 > (n_burn1 + n_burn2)) & (((g + 1 - n_burn1 - n_burn2) / thin - floor((g + 1 - n_burn1 - n_burn2) / thin)) == 0 )){
      
      iter = (g + 1 - n_burn1 - n_burn2)/thin - 1;
      
      // Lists
      // CLONE is important with list of lists
      Y_out[iter] = clone(Y_bis);
      X_out[iter] = clone(X_bis);
      rho_star_out[iter] = rho_star;
      theta0_star_out[iter] = theta0_star;
      psi0_star_out[iter] = psi0_star;
      nj_out[iter] = nj;
      gamma_X_out[iter] = gamma_X;
      
      // Cubes 
      theta_out.slice(iter) = theta;
      gamma_Y_out.slice(iter) = gamma_Y;
      
      // Matrices
      s_out.row(iter) = s.t();
      psi_out.row(iter) = psi.t();
      eta_theta_out.row(iter) = eta_theta.t();
      eta_psi_out.row(iter) = eta_psi.t();
      
      // Vectors
      if(update_sig2_eps_theta){
        sig2_eps_theta_out(iter) = sig2_eps_theta;
      }
      if(update_sig2_eps_psi){
        sig2_eps_psi_out(iter) = sig2_eps_psi;
      }
      K_I_out(iter) = K_I;
      
      kappa_out(iter) = kappa;
    }
    
    //Progress bar increment
    progr.increment(); 
  }
  
  //Print acceptance rates
  Rcout << "-- Accepptance Rates --\n";
  Rcout << "Subject parameters:\n";
  Rcout << "theta a.r. = " << theta_accept/n_tot << "\n";
  Rcout << "psi a.r. = " << psi_accept.t()/n_tot << "\n";
  
  Rcout << "Item parameters:\n";
  Rcout << "gamma_Y a.r. = " << gamma_Y_accept.t()/n_tot << "\n";
  for(int q = 0; q < QX; q++){
    arma::vec gamma_X_q_accept = gamma_X_accept[q];
    Rcout << "gamma_X_" << q << " a.r. = " << gamma_X_q_accept.t()/n_tot << "\n";
  }
  
  
  //Create lists outside. Max number of element is 20
  List YX_List = List::create(Named("Y_out") = Y_out, Named("X_out") = X_out);
  
  List theta_psi_List = List::create(Named("theta_out") = theta_out, Named("psi_out") = psi_out, Named("eta_theta_out") = eta_theta_out, Named("eta_psi_out") = eta_psi_out);
  
  List gamma_List;
  if(update_sig2_eps_theta){
    if(update_sig2_eps_psi){
      gamma_List = List::create(Named("gamma_Y_out") = gamma_Y_out, Named("gamma_X_out") = gamma_X_out, Named("sig2_eps_theta_out") = sig2_eps_theta_out, Named("sig2_eps_psi_out") = sig2_eps_psi_out);
    }else{
      gamma_List = List::create(Named("gamma_Y_out") = gamma_Y_out, Named("gamma_X_out") = gamma_X_out, Named("sig2_eps_theta_out") = sig2_eps_theta_out);
    }
  }else{
    if(update_sig2_eps_psi){
      gamma_List = List::create(Named("gamma_Y_out") = gamma_Y_out, Named("gamma_X_out") = gamma_X_out, Named("sig2_eps_psi_out") = sig2_eps_psi_out);
    }else{
      gamma_List = List::create(Named("gamma_Y_out") = gamma_Y_out, Named("gamma_X_out") = gamma_X_out);
    }
  }
  
  List BNP_List;
  BNP_List = List::create(Named("s_out") = s_out, Named("nj_out") = nj_out, Named("K_I_out") = K_I_out, Named("kappa_out") = kappa_out, Named("rho_star_out") = rho_star_out, Named("theta0_star_out") = theta0_star_out, Named("psi0_star_out") = psi0_star_out);
  
  return List::create(Named("YX_List") = YX_List, Named("theta_psi_List") = theta_psi_List, Named("gamma_List") = gamma_List, Named("BNP_List") = BNP_List);
}

