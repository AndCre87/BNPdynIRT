// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// BNPdynIRT_Gibbs
List BNPdynIRT_Gibbs(List data_list, List MCMC_list, List Param_list);
RcppExport SEXP _BNPdynIRT_BNPdynIRT_Gibbs(SEXP data_listSEXP, SEXP MCMC_listSEXP, SEXP Param_listSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type data_list(data_listSEXP);
    Rcpp::traits::input_parameter< List >::type MCMC_list(MCMC_listSEXP);
    Rcpp::traits::input_parameter< List >::type Param_list(Param_listSEXP);
    rcpp_result_gen = Rcpp::wrap(BNPdynIRT_Gibbs(data_list, MCMC_list, Param_list));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_BNPdynIRT_BNPdynIRT_Gibbs", (DL_FUNC) &_BNPdynIRT_BNPdynIRT_Gibbs, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_BNPdynIRT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
