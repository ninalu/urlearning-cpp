/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   adaptive_lasso_entropy.h
 * Author: nilu
 *
 * Created on June 24, 2017, 4:19 PM
 */

#ifndef ADAPTIVE_LASSO_ENTROPY_H
#define ADAPTIVE_LASSO_ENTROPY_H

#include <stdint.h>
#include <vector>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include "urlearning/base/typedefs.h"
#include "urlearning/base/bayesian_network.h"

#include "scoring_function.h"
#include "constraints.h"

#include <armadillo>

namespace scoring {

    class AdaptiveLassoEntropyScoringFunction : public ScoringFunction 
    {
    public:
        AdaptiveLassoEntropyScoringFunction(
                  datastructures::BayesianNetwork &network
                , int which
                , std::string data_file_name
                , double lambda
                , Constraints *constraints
                , bool is_adaptive
                , bool enableDeCamposPruning
                );

        ~AdaptiveLassoEntropyScoringFunction() {
            // no pointers 
        }

        float calculateScore(int variable, varset parents, FloatMap &cache);
        void post_processing(std::vector<int> & total_ordering, std::vector<varset>& optimalParents);
        void post_processing_one_parent(int variable, varset & parents);
        static double calculate_entropy(const arma::vec & error, const double orig_dev, int choice=0);
    private:
        float calculateScoreAndBeta(int variable, const varset & parents, const bool adaptive, arma::vec & beta, arma::uvec & parent_vec, int & num_parents);
        
        datastructures::BayesianNetwork network;
        const int which_score; // 0 for lasso, 1 for entropy1, 2 for entropy2
        const int variableCount;
        const int num_rows;// number of data points 
        const double lambda; // for L1 regularization
        const double lambda2; // for  L2 regularization
        Constraints *constraints;
        boost::unordered_set<varset> invalidParents;
        float baseComplexityPenalty;
        bool enableDeCamposPruning;
        const bool is_adaptive;
	bool beta_initialized_;
        arma::mat raw_data; // raw data in a column-major matrix
        arma::mat norm_data;// normalized data
        arma::mat corr_mat; // correlation matrix
        arma::mat beta_MLE_mat; // the matrix of beta from MLE
        arma::vec raw_data_mean;  // the mean of each column of raw_data
        arma::vec raw_data_dev;   // the sample deviation of raw data
        arma::vec raw_data_entropy;//the entropy of each column if considered independent
        arma::vec raw_data_lasso;//the lasso of each column if considered independent
        arma::vec raw_data_bic;//the bic of each column if considered independent
        std::string data_file_name;
        
    static const double pi;

    static const double k1;

    static const double k2a;

    static const double k2b;

    static const double Hnu;

    static const double root2overpi;

    static const double rootonehalf;
        
    };

}

#endif /* ADAPTIVE_LASSO_ENTROPY_H */

