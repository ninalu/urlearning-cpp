/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   BIC_OLS.cpp
 * Author: Ni Y Lu
 *
 * Created on June 24, 2017, 4:19 PM
 */


#include "BIC_OLS.h"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/core.hpp>
#include <cmath>
#include <iostream>
#include <unordered_set>


using namespace std;
using namespace mlpack;

namespace scoring
{

//constructor
BIC_OLS_Function::BIC_OLS_Function(
    datastructures::BayesianNetwork& network
  , std::string data_file_name_
  , scoring::Constraints *constraints
  , bool enableDeCamposPruning
  , double lambda_
  )
  : variableCount(network.size())
  , num_rows(0) // will get correct value at the end of constructor
  , lambda(lambda_)
  , verbose(false)
{
    this->network = network;
    this->constraints = constraints;
    this->enableDeCamposPruning = enableDeCamposPruning;
    data_file_name = data_file_name_;
    // read data
    //arma::mat raw_data0;
    mlpack::data::Load(data_file_name, raw_data, true, false); // fatal=true, transpose=true
    //raw_data=trans(raw_data0);
    cout << __FILE__ << ":" << __LINE__ << ", BIC constructor, Dimension of raw_data, rows " << raw_data.n_rows
         << ", cols " << raw_data.n_cols
	 << ", lambda " << lambda
         << endl;
    norm_data = raw_data;
    const_cast<int&>(variableCount) = raw_data.n_cols;
    const_cast<int&>(num_rows) = raw_data.n_rows;
    bic_threshold = 0.0 ; //(1.8-0.75*lambda)*log(num_rows);
    if(bic_threshold < 0)
	    bic_threshold = 0.0;
    raw_data_mean = arma::vec(variableCount);
    raw_data_dev = arma::vec(variableCount);
    raw_data_entropy = arma::vec(variableCount);
    raw_data_lasso   = arma::vec(variableCount);
    raw_data_bic     = arma::vec(variableCount);
    
    for(int i=0; i < variableCount; i++)
    {
        arma::vec x = raw_data.col(i);
        double mean_x = mean(x);
        x -= mean_x;
        double dev_x = sqrt(var(x));
        double dev_x2= dev_x; 
        //normalize data
        for(int j=0; j<num_rows;j++)
        {
            raw_data(j,i)  = raw_data(j,i) - mean_x ;
            norm_data(j,i) = raw_data(j,i) / dev_x2 ; // normalize by dev_x2 = sqrt(num_rows)*dev_x, to avoid adjusting lambda with n_rows
        }
        raw_data_mean(i) = mean_x; 
        raw_data_dev(i) = dev_x;

        double lasso_score = 0;
        for(int j=0; j<num_rows;j++)
            lasso_score += x(j)*x(j);
	raw_data_lasso(i) = lasso_score;
	//raw_data_bic(i) = num_rows*log(lasso_score/num_rows/dev_x/dev_x) + log(num_rows); // k=1 for a single variance
	raw_data_bic(i) = 0.0; // first term zero because of unit variance. k=1 for a single variance
        cout << __FILE__ << ":" << __LINE__ <<  ", BIC ctor, Raw data for variable #" << i 
             << ", mean " << mean_x 
             << ", dev " << dev_x 
             << ", dev2 "<< dev_x2 << ", dev2/dev = " << dev_x2/dev_x
             << ", sqrtN = " << sqrt(num_rows)
             << ", num_rows " << num_rows
             << ", variableCount " << variableCount
	     << ", raw_data_bic(" << i << ")= " << raw_data_bic(i)
             << endl;
    }
    // Now calculate the MLE beta matrix
    arma::vec beta;
    arma::uvec parent_vec = arma::uvec(variableCount);

    VARSET_NEW(parents, variableCount);
    VARSET_SET_ALL(parents, variableCount);

    corr_mat = trans(norm_data) * norm_data / norm_data.n_rows;
    printf("Printing correlation_mat\n");
    printf("%6c", ' ');
    for(int i=0; i<variableCount; i++)
        printf(" var %02d", i+1);
    printf("\n");
    for(int i=0;i < variableCount; i++)
    {
        printf("Row %2d", i+1);
        for(int j=0; j<variableCount; j++)
              printf("%7.3f",corr_mat(i,j));
        printf("\n");
    }

    printf("%6c", ' ');
    for(int i=0; i<variableCount; i++)
        printf(" var %02d", i+1);
    printf("\n");
}// constructor BIC_OLS_Function

float BIC_OLS_Function::find_best_subset_score( const varset & parents
                                              , FloatMap &cache
                                              , const arma::uvec & parent_vec
                                              , const int num_parents
                                              , std::unordered_set<varset> & checked
                                              , varset & optimal_subset // return the optimal subset to caller 
                                              )
{
    float best_score = 0;
    for(int idx = 0; idx < num_parents; idx++)
    {
      const int varIdx = parent_vec[idx];
      VARSET_COPY(parents, thin_parents);
      VARSET_CLEAR(thin_parents, varIdx);
      // if already checked, if thin_parents is empty set, then it will stop here
      if(checked.find(thin_parents) != checked.end()) 
	continue; 
      auto iter = cache.find(thin_parents);
      if(iter != cache.end()) 
      {
         if(iter->second > best_score)
	 {
	   best_score = iter->second;
	   optimal_subset = iter->first;
	 }
      }
      else // check smaller subsets
      {
        arma::uvec  new_parent_vec = arma::uvec(num_parents-1);
	int j=0;
	for(int i=0; i<num_parents; i++)
	{
	  if(varIdx == int(parent_vec[i]) ) 
	    continue;
          new_parent_vec[j++] = parent_vec[i];
	  VARSET_COPY(parents, the_subset);
	  float the_score = find_best_subset_score(thin_parents, cache, new_parent_vec, num_parents-1, checked, the_subset);
	  checked.insert(thin_parents);
	  if(the_score > best_score)
	  {
	    best_score = the_score;
	    optimal_subset = the_subset;
	  }
	}// for i
      }//end else
    }//for idx
    return best_score;
}

float BIC_OLS_Function::calculateScore(int variable, varset parents, FloatMap &cache)
{
    //const double bic_threshold = log(num_rows);
    arma::vec beta;
    arma::uvec parent_vec = arma::uvec(variableCount);
    parent_vec.fill(0);
    varset orig_parents0 = parents;
    const int orig_cache_size = cache.size();
//  store neighbor index into a uvec
    int num_parents = 0;

    for(int i=0; i<variableCount; i++)
    {
        if(i != variable and VARSET_GET(parents, i))
        {
            parent_vec(num_parents++) = i;
        }
    }
// check whether a smaller parent set exist, otherwise just skip this parent set
/*
    for(int i=0; num_parents > 7 and i < num_parents; i++)
    {
        int parent_index = parent_vec(i);
	varset thin_parents = parents;
        VARSET_CLEAR(thin_parents, parent_index);
        FloatMap::const_iterator iter = cache.find(thin_parents);
        if(iter == cache.end())
	{
          // This is where we can prune by recording unrelated parent_index
          cout << __FILE__ << ":" << __LINE__
           << ", not accepting spurious parent before regression, variable index " << variable
           << ", spurious parent set " << varsetToString(parents)
           << ", parent_index " << parent_index
           << std::endl;
          return 0.0;
	}
    }
    */
    float the_score = calculateScoreAndBeta(variable, parents, beta, parent_vec, num_parents);
    if( num_parents > 0 and the_score >= bic_threshold ) // skipping bad (not so negative) score
    {
	if(verbose)
        cout << __FILE__ << ":" << __LINE__
             << ", not accepting spurious parent after regression, BIC too close to zero, variable index " << variable
             << ", num_parents " << num_parents
	     << ", the_score " << the_score
	     << ", bic_threshold " << bic_threshold
             << ", spurious parent set " << varsetToString(parents)
             << std::endl;
	return -the_score;
    }
    // check whether some beta's are zero
    //cout << "Clearing zero beta's bits for variable " << variable;
    int k=0;
    int skips = 0;
    VARSET_NEW(empty_set, variableCount);
    std::unordered_set<varset> checked;
    checked.insert(empty_set); // insert empty set
    VARSET_COPY(parents, subset);
    float best_subset_score = find_best_subset_score(parents, cache, parent_vec, num_parents, checked, subset );
    if(num_parents > 0 and best_subset_score + bic_threshold >= -the_score)
    {
      if(verbose)
      cout << __FILE__ << ":" << __LINE__
               << ", not accepting extra spurious parent, variable index " << variable
               << ", spurious parent set " << varsetToString(parents)
	       << ", optimal subset " << varsetToString(subset)
               << ", the_score " << -the_score
	       << ", best_subset_score " << best_subset_score
               << std::endl
               ;
      return -the_score;
    }

    // save into cache
    cache[parents] = -the_score;

    if(verbose)
    cout << __FILE__ << ":" << __LINE__ 
         << ", CalculateScore,  Actual parameters k = " << k
	 << ", variable " << variable 
	 << ", num_parents " << num_parents
	 << ", orig parents " << std::hex << orig_parents0 
	 << ", new parents " << parents << std::dec
	 << ", optimal subset " << subset
	 << ", the_score " << -the_score
	 << ", cache sizes " << orig_cache_size
	 << " -> " << cache.size()
	 << endl;
    //cout << endl;
    //FloatMap::const_iterator iter = cache.find(parents);
    //if( iter == cache.end() or iter->second < the_score)
    /*
    else 
    {
        cout << "Variable # " << variable << "already has parents " << std::hex << parents << std::dec
             << " with neg score  " << iter->second
             << ", ignored new neg value " << -the_score
             << endl;
    }
    */
    return -the_score;
}
float BIC_OLS_Function::calculateScoreAndBeta(int variable, const varset & parents, arma::vec & beta, arma::uvec & parent_vec, int & num_parents)
{
    // check if this violates the constraints
    if (constraints != NULL && !constraints->satisfiesConstraints(variable, parents)) {
        invalidParents.insert(parents);
        return variableCount;// trying to return a big number here, but not too big to cause overflow
    }
    
    //arma::mat X(num_rows,1);
    //X.fill(1.0);
    num_parents = 0;

    for(int i=0; i<variableCount; i++)
    {
        if(i != variable and VARSET_GET(parents, i))
        {
            parent_vec(num_parents++) = i;
            //X.insert_cols(num_parents++, raw_data.col(i));
            //cout<<X[i]<<endl;
        }
    }

    // Y mustbe a rowvec because mlpack LinearRegression expects Y to be a rowvec
    arma::rowvec Y=trans(norm_data.col(variable));

    if(0 == num_parents )
    {
        return 0.0;
    }

    arma::mat X=trans(norm_data.cols(parent_vec.head(num_parents)) );

    double varY = sqrt(var(Y));
    double varX0= sqrt(var(X.col(num_parents - 1)));
    //regression::LARS lasso1(true, lambda);
    //lasso1.Train(X,Y,beta,false);// use true if X is column-major, and false if X is row-major
    regression::LinearRegression lr(X, Y, 0, false);
    beta = lr.Parameters();
    double error_L2 = lr.ComputeError(X,Y); 
    const int num_beta = beta.n_rows > beta.n_cols ? beta.n_rows : beta.n_cols ;
    int k=0;
    for(int i=0; i < num_beta ; i++)
    {
        k += abs(beta(i)) > 1e-8;
    }
    if(verbose)
    cout<<"LinearRegression result, beta.n_rows = "<< beta.n_rows
	<< ", beta.n_cols= " << beta.n_cols
        << ", num_parents = " << num_parents
	<< ", non-zero parents = " << k
        << ", X.n_rows = " << X.n_rows
        << ", X.n_cols = " << X.n_cols
        << ", devY " << varY
        << ", devX0  " << varX0
	<< ", intercept = " << lr.Intercept()
        <<endl;
    /*
    cout << "Before running Yhat, X dimension (" << X.n_rows
	 << ", " << X.n_cols 
	 << "), beta dimension(" << beta.n_rows
	 << ", " << beta.n_cols 
	 << "), Y dimension (" << Y.n_rows 
	 << ", " << Y.n_cols << ")"
	 << ", error_L2 = " << error_L2 
	 << ", intercept = " << lr.Intercept()
	 << std::endl;
	 */

    double sum_error_sq = 0;
    arma::rowvec Yhat = trans(beta)*X; //raw_data_mean(variable);
    arma::rowvec error = Yhat - Y;
    const int num_err = error.n_rows > error.n_cols ? error.n_rows : error.n_cols ;
    for(int i=0; i < num_err ; i++)
        sum_error_sq += error(i)*error(i);
    /*
    //Yhat.fill(0.0);
    cout << "After running Yhat, Yhat dimension (" << Yhat.n_rows
	 << ", " << Yhat.n_cols 
	 << "), Y dimension (" << Y.n_rows
	 << ", " << Y.n_cols
	 << "), beta dim(" << beta.n_rows
	 << ", " << beta.n_cols 
	 << ")"
	 << std::endl;
    const int num_err = error.n_rows > error.n_cols ? error.n_rows : error.n_cols ;
    for(int i=0; i < num_err ; i++)
        sum_error_sq += error(i)*error(i);
    */

    double the_score = num_err*log(error_L2) + lambda*log(num_err)*num_parents - raw_data_bic(variable) ;
    double error_L2sq = num_err*error_L2;
//    double the_score = error_L2sq + 2*lambda*log(num_err)*num_parents - raw_data_bic(variable) ;
   // double the_score = sum_error_sq + lambda*log(num_err)*num_parents - raw_data_bic(variable) ;
   // double the_score = num_err*log(sum_error_sq/num_err) + lambda*log(num_err)*num_parents - raw_data_bic(variable) ;

    if(verbose)
    cout << "Calculated BIC score for variable " << variable
         << ", the_score " << the_score
         << ", independent_score " << raw_data_bic(variable)
         << ", parents " << varsetToString(parents)
         << ",  num_beta " << num_beta
         << ", beta dim("    << beta.n_rows << ", " << beta.n_cols
         << "), Y.n_rows = " << Y.n_rows
         << ", num_err = " << num_err
         << ", k = " << k
	 << ", error_L2sq = " << error_L2sq
	 << ", sum_error_sq = " << sum_error_sq
	 << ", ratio = " << (sum_error_sq/error_L2sq)
	 << ", intercept = " << lr.Intercept()
         << endl
         ; 
    return the_score;
}
void BIC_OLS_Function::post_processing_one_parent(int variable, varset & parents)
{
    arma::vec beta;
    arma::uvec parent_vec = arma::uvec(variableCount);
    parent_vec.fill(0);
    int num_parents = 0;
    float the_score = calculateScoreAndBeta(variable, parents, beta, parent_vec, num_parents);
    cout << "Post processing, variable " << (1+variable)
         << ", score " << the_score << ", " << num_parents << " coefficients"
         ;
    for(int j=0; j<num_parents; j++)
    {
//        if(abs(beta(j)) < 1e-200)
//            continue;
        int col = parent_vec(j);
        cout << ", (" << (1+col) << ", " << beta(j) 
//             << ", " << raw_data_dev(col)
             << ")";
    }
    cout << endl;
}
void BIC_OLS_Function::post_processing(std::vector<int> & total_ordering, std::vector<varset>& optimalParents)
{
    // run lasso for post processing
    arma::vec beta;
    arma::uvec parent_vec = arma::uvec(variableCount);
    parent_vec.fill(0);
    VARSET_NEW(parents, variableCount);
    VARSET_CLEAR_ALL(parents);
    for(int i=0; i < variableCount; i++)
    {
        post_processing_one_parent(total_ordering[i], parents);
        post_processing_one_parent(total_ordering[i], optimalParents[i]);
        VARSET_SET(parents, total_ordering[i]);
    }
}

}//end namespace scoring
