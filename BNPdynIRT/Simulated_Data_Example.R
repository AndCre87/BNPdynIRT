# Analysis of multiple questionnaire data via BNPdynIRT model
# Model described in Cremaschi et al. 2021
# Simulated data for two sets of reporters
# The first set is observed at different time points (TY), the second one for several (QX) questionnaires
# We include an AR term and DP prior for clustering, as well as covariates

rm(list = ls())
set.seed(123)

#Load R package
library("BNPdynIRT")

### Number of subjects
I <- 200

TY <- 3 #Number of time points for first respondent group
JY <- c(25, 20, 30) #Number of questions at each time point
mY <- 3 #Number of categories for each question

max_J <- 40 #Number of unique questions
#Indices of question from 1 to max_J (total number of unique questions)
qn_index <- list()
qn_index[[1]] <- c(1:JY[1]) #the first JY[1] questions are relative to time 1
qn_index[[2]] <- c(1:JY[2]) #the first JY[2] questions are relative to time 2
qn_index[[3]] <- c(1:15, 26:40) #Last time point has only some questions in common in different order


QX <- 2 #Number of questionnaires for second respondent group
JX <- c(20, 30) #Number of questions in each questionnaire
mX <- c(3, 4) #Number of categories for each question

### Simulate data by fixing item parameters ###
#Discriminatory parameters are from a mixture with two components (0.2, 0.8), of which the first one is highly discriminatory
alpha_Y_simul <- rep(0, max_J)
alpha_Y_simul_mean <- c(1, 2)
beta_Y_simul <- matrix(0, max_J, mY)
for(j in 1:max_J){
  ind_j <- runif(1) < 0.8
  alpha_Y_simul[j] <- rnorm(1, alpha_Y_simul_mean[ind_j + 1], 1)
  beta_Y_simul[j,] <- rnorm(mY, 0, 1)
  beta_Y_simul[j,1] <- -sum(beta_Y_simul[j,2:mY]) #Apply constraints to beta's
}
alpha_X_simul <- list()
alpha_X_simul_mean <- c(1, 2)
beta_X_simul <- list()
for(iq in 1:QX){
  alpha_X_simul_iq <- rep(0, JX[iq])
  beta_X_simul_iq <- matrix(0, JX[iq], mX[iq])
  for(j in 1:JX[iq]){
    ind_j <- runif(1) < 0.8
    alpha_X_simul_iq[j] <- rnorm(1, alpha_X_simul_mean[ind_j + 1], 1)
    beta_X_simul_iq[j,] <- rnorm(mX[iq], 0, 1)
    beta_X_simul_iq[j,1] <- -sum(beta_X_simul_iq[j,2:mX[iq]]) #Apply constraints to beta's
  }
  alpha_X_simul[[iq]] <- alpha_X_simul_iq
  beta_X_simul[[iq]] <- beta_X_simul_iq
}


#Simulate covariates
qY <- 3
UY <- array(NA, dim = c(I, TY, qY))
cat_cov <- runif(I) < 0.25 #Categorical covariate
cont_cov <- rnorm(I) #Continuous time-homogeneous covariate
UY[,,1] <- matrix(cat_cov, I, TY) 
UY[,,2] <- matrix(cont_cov, I, TY) 
UY[,,3] <- matrix(rgamma(I, 1, 1), I, TY, byrow = TRUE) #Time-varying covariate

qX <- 2
UX <- matrix(NA, I, qX)
UX[,1] <- matrix(cat_cov, I, 1) #Categorical covariate
UX[,2] <- matrix(cont_cov, I, 1) #Continuous time-homogeneous covariate

#Simulate latent subject profiles 

#two clusters
K_I_simul <- 2
s_simul <- sample.int(2, I, replace = TRUE, prob = c(1/3,2/3))

rho_star_simul <- -1 + 2 * runif(K_I_simul)
theta0_star_simul <- rnorm(K_I_simul)
psi0_star_simul <- rnorm(K_I_simul)

eta_theta_simul <- c(1, -1, 0) #three covariates
eta_psi_simul <- c(1, -0.5) #two covariates

sig2_eps_psi_simul <- 0.5
sig2_eps_theta_simul <- 0.5

psi_simul <- rnorm(psi0_star_simul[s_simul] + UX %*% eta_psi_simul, sd = sqrt(sig2_eps_psi_simul))
theta_simul <- matrix(NA, I, TY)
for(i in 1:I){
  theta_simul[,1] <- rnorm(theta0_star_simul[s_simul] + UY[,1,] %*% eta_theta_simul, sd = sqrt(sig2_eps_theta_simul))
  for(t in 2:TY){
    theta_simul[,t] <- rnorm(theta0_star_simul[s_simul] + rho_star_simul[s_simul] * theta_simul[,t-1] + UY[,t,] %*% eta_theta_simul, sd = sqrt(sig2_eps_theta_simul))
  }
}



# Simulate of responses
Y_simul <- vector("list", length = TY)
for(t in 1:TY){
  Y_simul_t <- matrix(0, I, JY[t])
  for(i in 1:I){
    for(j in 1:JY[t]){
      ind_j <- qn_index[[t]][j]
      
      alpha_Y_j <- alpha_Y_simul[ind_j]
      beta_Y_j_cumsum <- cumsum(beta_Y_simul[ind_j,])
      
      probs_t <- rep(0, mY)
      for(l in 1:mY){
        probs_t[l] <- alpha_Y_j * ((l-1) * theta_simul[i,t] - beta_Y_j_cumsum[l])
      }
      probs_t <- exp(probs_t - max(probs_t)) / sum(exp(probs_t - max(probs_t)))
      
      Y_simul_t[i,j] <- sample.int(n = mY, size = 1, prob = probs_t)
    }
  }
  Y_simul[[t]] <- Y_simul_t - 1 #start from 0 in Rcpp
}


X_simul <- vector("list", length = QX)
for(iq in 1:QX){
  X_simul_iq <- matrix(0, I, JX[iq])
  for(j in 1:JX[iq]){
    for(i in 1:I){
      alpha_X_j <- alpha_X_simul[[iq]][j]
      beta_X_j_cumsum <- cumsum(beta_X_simul[[iq]][j,])
      
      probs_iq <- rep(0, mX[iq])
      for(l in 1:mX[iq]){
        probs_iq[l] <- alpha_X_j * ((l-1) * psi_simul[i] - beta_X_j_cumsum[l])
      }
      probs_iq <- exp(probs_iq - max(probs_iq)) / sum(exp(probs_iq - max(probs_iq)))
      
      X_simul_iq[i,j] <- sample.int(n = mX[iq], size = 1, prob = probs_iq)
    }
  }
  X_simul[[iq]] <- X_simul_iq - 1 #start from 0 in Rcpp
}

#Remove 1 from question indices (for Rcpp)
for(t in 1:TY){
  qn_index[[t]] <- qn_index[[t]] - 1
}


data_list <- list(Y = Y_simul, UY = UY, qn_index = qn_index, X = X_simul, UX = UX)


#ALgorithm specification
n_burn1 <- 100
n_burn2 <- 10000
thin <- 1
n_save <- 5000

#Initial partition
s_init = rep(0, I)
MCMC_list <- list(n_burn1 = n_burn1, n_burn2 = n_burn2, n_save = n_save, thin = thin, s_init = s_init)

## Priors and hyperparameters ##


#Apply constraints on betas
fix_betas <- TRUE
if(fix_betas){
  mY_p1 <- mY
  mX_p1 <- mX
}else{
  mY_p1 <- mY + 1
  mX_p1 <- mX + 1
}
#Prior for coefficients of prior of item parameters
mu_gamma_Y <- rep(0, mY_p1)
Sigma_gamma_Y <- diag(mY_p1)

mu_gamma_X <- vector("list", length = QX)
Sigma_gamma_X <- vector("list", length = QX)
for(iq in 1:QX){
  mu_gamma_X[[iq]] <- rep(0, mX_p1[iq])
  Sigma_gamma_X[[iq]] <- diag(mX_p1[iq])
}

#Prior on regression coefficients of subject parameters
m_eta_theta <- rep(0, qY)
V_eta_theta <- diag(qY)
m_eta_psi <- rep(0, qX)
V_eta_psi <- diag(qX)

#Prior on variances of latent trait parameters
update_sig2_eps_theta <- TRUE
sig2_eps_theta <- 1
a_sig2_eps_theta <- 3
b_sig2_eps_theta <- 2

update_sig2_eps_psi <- TRUE
sig2_eps_psi <- 1
a_sig2_eps_psi <- 3
b_sig2_eps_psi <- 2


#Choose process: DP(1) or NGG(2) or GDP(3)
# Centering measure P0
a_rho <- 1
b_rho <- 1
m_theta0 <- 0
sig2_theta0 <- 1
m_psi0 <- 0
sig2_psi0 <- 1

#For DP fix kappa
update_kappa <- FALSE
s2 <- 1
kappa <- 1
a_kappa <- 1
b_kappa <- 1

#Do we update the allocations?
update_s <- TRUE
process_P <- list(update_s = update_s, update_kappa = update_kappa, kappa = kappa, a_kappa = a_kappa, b_kappa = b_kappa)


Param_list <- list(process_P = process_P, fix_betas = fix_betas,
                   mu_gamma_Y = mu_gamma_Y, mu_gamma_X = mu_gamma_X, Sigma_gamma_Y = Sigma_gamma_Y, Sigma_gamma_X = Sigma_gamma_X,
                   m_eta_theta = m_eta_theta, V_eta_theta = V_eta_theta, m_eta_psi = m_eta_psi, V_eta_psi = V_eta_psi,
                   update_sig2_eps_theta = update_sig2_eps_theta, update_sig2_eps_psi = update_sig2_eps_psi, sig2_eps_theta = sig2_eps_theta, sig2_eps_psi = sig2_eps_psi,
                   a_sig2_eps_theta = a_sig2_eps_theta, b_sig2_eps_theta = b_sig2_eps_theta, a_sig2_eps_psi = a_sig2_eps_psi, b_sig2_eps_psi = b_sig2_eps_psi, 
                   a_rho = a_rho, b_rho = b_rho, m_theta0 = m_theta0, sig2_theta0 = sig2_theta0, m_psi0 = m_psi0, sig2_psi0 = sig2_psi0)

set.seed(8)
MCMC_output <- BNPdynIRT_Gibbs(data_list = data_list, MCMC_list = MCMC_list, Param_list = Param_list)
#every 100 iterations it print the number of clusters, the cluster sizes, and the ranges of the latent theta and psi
#Prints all the acceptance rates at the end

# save(MCMC_output, file = "OUTPUT_BNPdynIRT.RData")



### Some plots ###
library("fields")
library("ggplot2")
library("ggpubr")


#Traceplots of subject and item parameters

### Subject parameters
theta_psi_List <- MCMC_output$theta_psi_List

#theta
theta_out <- theta_psi_List$theta_out
matplot(t(theta_out[,1,]), type = "l")
matplot(t(theta_out[,2,]), type = "l")
matplot(t(theta_out[,3,]), type = "l")

#psi
psi_out <- theta_psi_List$psi_out
matplot(psi_out, type = "l")

### Item parameters
gamma_List <- MCMC_output$gamma_List

#sig2
sig2_eps_theta_out <- gamma_List$sig2_eps_theta_out
plot(sig2_eps_theta_out, type = "l")
abline(h = sig2_eps_theta_simul, col = "red")

sig2_eps_psi_out <- gamma_List$sig2_eps_psi_out
plot(sig2_eps_psi_out, type = "l")
abline(h = sig2_eps_psi_simul, col = "red")


#gamma_Y
gamma_Y_out <- gamma_List$gamma_Y_out

j <- 10
gamma_Y_j_out <- matrix(0, n_save, mY + 1)
for(g in 1:n_save){
  gamma_Y_j_out[g,] <- gamma_Y_out[j,,g]
}
matplot(gamma_Y_j_out, type = "l", lty = 1)
abline(h = log(alpha_Y_simul[j]), col = "red", lwd = 2)
abline(h = beta_Y_simul[j,], col = "red", lwd = 2)
if(fix_betas){
  print(rowSums(gamma_Y_j_out[,-1]))
}


#gamma_X
gamma_X_out <- gamma_List$gamma_X_out

q <- 2
j <- 10
gamma_X_q_j_out <- matrix(0, n_save, mX[q] + 1)
for(g in 1:n_save){
  gamma_X_q_j_out[g,] <- as.matrix(gamma_X_out[[g]][[q]][j,])
}
matplot(gamma_X_q_j_out, type = "l", lty = 1)
abline(h = log(alpha_X_simul[[q]][j]), col = "red", lwd = 2)
abline(h = beta_X_simul[[q]][j,], col = "red", lwd = 2)
if(fix_betas){
  print(rowSums(gamma_X_q_j_out[,-1]))
}


### BNP part
BNP_List <- MCMC_output$BNP_List
s_out <- BNP_List$s_out + 1

n_i <- 5
ind_i <- sample.int(I, n_i)
#rho
rho_star_out <- BNP_List$rho_star_out
rho_i_out <- matrix(0, n_save, n_i)
for(i in 1:n_i){
  for(g in 1:n_save){
    rho_i_out[g,i] <- rho_star_out[[g]][s_out[g,ind_i[i]]]
  }
}
matplot(rho_i_out, type = "l")

#theta0
theta0_star_out <- BNP_List$theta0_star_out
theta0_i_out <- matrix(0, n_save, n_i)
for(i in 1:n_i){
  for(g in 1:n_save){
    theta0_i_out[g,i] <- theta0_star_out[[g]][s_out[g,ind_i[i]]]
  }
}
matplot(theta0_i_out, type = "l")

#psi0
psi0_star_out <- BNP_List$psi0_star_out
psi0_i_out <- matrix(0, n_save, n_i)
for(i in 1:n_i){
  for(g in 1:n_save){
    psi0_i_out[g,i] <- psi0_star_out[[g]][s_out[g,ind_i[i]]]
  }
}
matplot(psi0_i_out, type = "l")




#Number of clusters
K_I_out <- BNP_List$K_I_out
plot(K_I_out, type = "l", lwd = 2, xlab = bquote(K[I]), ylab = "", main = "")
plot(table(K_I_out)/n_save, lwd = 5, xlab = bquote(K[I]), ylab = "", main = "", col = "blue", cex.lab = 3)
abline(v = K_I_simul, col = "red")




## Distribution of regression coefficients: indicate significance with 95% CI
theta_psi_List <- MCMC_output$theta_psi_List

#eta_theta
eta_theta_out <- theta_psi_List$eta_theta_out
eta_theta_mean <- colMeans(eta_theta_out)
eta_theta_CI <- apply(eta_theta_out, 2, quantile, probs = c(0.025, 0.975))
eta_theta_sig <- (eta_theta_CI[1,] > 0) + (eta_theta_CI[2,] < 0)

#eta_psi
eta_psi_out <- theta_psi_List$eta_psi_out
eta_psi_mean <- colMeans(eta_psi_out)
eta_psi_CI <- apply(eta_psi_out, 2, quantile, probs = c(0.025, 0.975))
eta_psi_sig <- (eta_psi_CI[1,] > 0) + (eta_psi_CI[2,] < 0)

eta_psi_theta_df <- data.frame(x_mean = c(eta_theta_mean, eta_psi_mean),
                               index = as.factor(c(c(1:qY), c(1:qX))),
                               CI_low = c(eta_theta_CI[1,], eta_psi_CI[1,]),
                               CI_upp = c(eta_theta_CI[2,], eta_psi_CI[2,]),
                               YX = as.factor(c(rep(1, qY), rep(2, qX))),
                               Significance = as.factor(c(eta_theta_sig, eta_psi_sig)))

eta_names <- c("X1", "X2", "X3", "X1", "X2")

labs_plot <- c("1" = "Mothers", "2" = "Children")

print(ggplot(eta_psi_theta_df, aes(x = x_mean, y = index, col = Significance)) + 
        geom_errorbar(aes(xmin = CI_low, xmax = CI_upp), width = 0.25, col = "blue") +
        geom_point(size = 2) +
        geom_vline(xintercept = 0, linetype = "dashed", size = 0.5) +
        scale_color_manual(values = c("blue", "red")) +
        scale_y_discrete(breaks = as.factor(c(c(1:qY), c(1:qX))), labels = eta_names) +
        labs(x = "", y = "", title = bquote("Posterior means of "~(eta^theta~","~eta^psi)~" and 95% CI")) +
        facet_grid(cols = vars(YX), scales = "free", labeller = as_labeller(labs_plot)) +
        theme_bw() +
        theme(axis.title = element_text(size = 15), plot.title = element_text(size = 15, face = "bold"), axis.text = element_text(size=15), legend.position = "none")
)

