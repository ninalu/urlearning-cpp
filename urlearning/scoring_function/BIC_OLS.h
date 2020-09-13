/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   BIC_OLS.h
 * Author: Ni Y Lu
 *
 * Created on June 24, 2017, 4:19 PM
 */

#ifndef CONTINUOUS_OLS_BIC_H
#define CONTINUOUS_OLS_BIC_H

#include <stdint.h>
#include <vector>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include "urlearning/base/typedefs.h"
#include "urlearning/base/bayesian_network.h"

#include "scoring_function.h"
#include "constraints.h"

#include <armadillo>
#include <unordered_set>

namespace scoring {

    class BIC_OLS_Function : public ScoringFunction 
    {
    public:
        BIC_OLS_Function(
                  datastructures::BayesianNetwork &network
                , std::string data_file_name
                , Constraints *constraints
                , bool enableDeCamposPruning
		, double lambda
                );

        ~BIC_OLS_Function() {
            // no pointers 
        }

        float calculateScore(int variable, varset parents, FloatMap &cache);
        void post_processing(std::vector<int> & total_ordering, std::vector<varset>& optimalParents);
        void post_processing_one_parent(int variable, varset & parents);
    private:
        float calculateScoreAndBeta(int variable, const varset & parents, arma::vec & beta, arma::uvec & parent_vec, int & num_parents);
	static float find_best_subset_score( const varset & parents
                                           , FloatMap &cache
                                           , const arma::uvec & parent_vec
                                           , const int num_parents
                                           , std::unordered_set<varset> & checked
                                           , varset & optimal_subset // return the optimal subset to caller
                                           );

        
        datastructures::BayesianNetwork network;
        const int variableCount;
        const int num_rows;// number of data points 
        Constraints *constraints;
        boost::unordered_set<varset> invalidParents;
        float baseComplexityPenalty;
        bool enableDeCamposPruning;
        arma::mat raw_data; // raw data in a column-major matrix
        arma::mat norm_data;// normalized data
        arma::mat corr_mat; // correlation matrix
        arma::vec raw_data_mean;  // the mean of each column of raw_data
        arma::vec raw_data_dev;   // the sample deviation of raw data
        arma::vec raw_data_entropy;//the entropy of each column if considered independent
        arma::vec raw_data_lasso;//the lasso of each column if considered independent
        arma::vec raw_data_bic;//the bic of each column if considered independent
        std::string data_file_name;
	const double lambda;
	double bic_threshold;
	const bool verbose;
    };

}

#endif /* ADAPTIVE_LASSO_ENTROPY_H */

