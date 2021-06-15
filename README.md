# BNPdynIRT
RCpp package for the implementation of a Bayesian semiparametric dynamic IRT model

From the manuscript: A Bayesian nonparametric approach to dynamic item-response modelling: an application to the GUSTO cohort study (2021)

# Authors
Andrea Cremaschi, Singapore Institute for Clinical Sciences, A*STAR, Singapore.

# Description
Statistical analysis of questionnaire data is often performed employing techniques from item-response theory (IRT). In this framework, it is possible to differentiate respondent profiles and characterize the questions (items) included in the questionnaire via interpretable parameters. These models are often cross-sectional and aim at evaluating the performance of the respondents. Building on the current literature, we propose a Bayesian semiparametric model and extend the current literature by: (i) introducing temporal dependence among questionnaires taken at different time points; (ii) jointly modelling the responses to questionnaires taken from different, but related, groups of subjects (in our case mothers and children), introducing a further dependency structure and therefore sharing of information; (iii) allowing clustering of subjects based on their latent response profile.

# Contents
BNPdynIRT: R package using Rcpp-Armadillo libraries for the implementation
Simulated_Data_Example.R: An example of implementation to simulated data.
