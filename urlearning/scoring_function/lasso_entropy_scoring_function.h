/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   lasso_entropy_scoring_function.h
 * Author: nilu
 *
 * Created on June 7, 2017, 5:06 AM
 */

#ifndef LASSO_ENTROPY_SCORING_FUNCTION_H
#define LASSO_ENTROPY_SCORING_FUNCTION_H

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

    class LassoEntropyScoringFunction : public ScoringFunction 
    {
    public:
        LassoEntropyScoringFunction(
                  datastructures::BayesianNetwork &network
                , int which
                , std::string data_file_name
                , double lambda
                , Constraints *constraints
                , bool enableDeCamposPruning
                );

        ~LassoEntropyScoringFunction() {
            // no pointers 
        }

        float calculateScore(int variable, varset parents, FloatMap &cache);
        static double calculate_entropy(const arma::vec & error, const double orig_dev, int choice=0);
    private:

        datastructures::BayesianNetwork network;
        const int which_score; // 0 for lasso, 1 for entropy1, 2 for entropy2
        const int variableCount;
        const int num_rows;// number of data points 
        const double lambda;
        Constraints *constraints;
        boost::unordered_set<varset> invalidParents;
        float baseComplexityPenalty;
        bool enableDeCamposPruning;
        arma::mat raw_data; // raw data in a column-major matrix
        arma::mat norm_data;// normalized data
        arma::vec raw_data_mean;  // the mean of each column of raw_data
        arma::vec raw_data_dev;   // the sample deviation of raw data
        arma::vec raw_data_entropy;//the entropy of each column if considered independent
        std::string data_file_name;
        
    };

}


#endif /* LASSO_ENTROPY_SCORING_FUNCTION_H */

