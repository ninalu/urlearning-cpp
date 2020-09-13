/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/*
 * File:   adaptive_lasso_entropy.cpp
 * Author: Ni Y Lu
 *
 */


#include "adaptive_lasso_entropy.h"
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/core.hpp>
#include <cmath>
#include <iostream>


using namespace std;
using namespace mlpack;

namespace scoring
{
const double AdaptiveLassoEntropyScoringFunction::pi = M_PI;

const double AdaptiveLassoEntropyScoringFunction::k1 = 36.0/(8*sqrt(3) - 9.0);

const double AdaptiveLassoEntropyScoringFunction::k2a = 1.0/(2.0-6.0/pi);

const double AdaptiveLassoEntropyScoringFunction::k2b = 24/(16.0*sqrt(3)-27.0);

//const double AdaptiveLassoEntropyScoringFunction::Hnu = 0.5*(1.0+log(2.0*pi));
const double AdaptiveLassoEntropyScoringFunction::Hnu = 1.0/(1.0+log(2.0*pi));

const double AdaptiveLassoEntropyScoringFunction::root2overpi = sqrt(2.0/pi);

const double AdaptiveLassoEntropyScoringFunction::rootonehalf = sqrt(0.5);
    
double AdaptiveLassoEntropyScoringFunction::calculate_entropy(const arma::vec & x, const double original_dev, int choice)

{
//    const double pi = M_PI;
//
//    const double k1 = 36.0/(8*sqrt(3) - 9.0);
//
//    const double k2a = 1.0/(2.0-6.0/pi);
//
//    const double k2b = 24/(16.0*sqrt(3)-27.0);
//
//    const double Hnu = 0.5*(1.0+log(2.0*pi));
//
//    const double root2overpi = sqrt(2.0/pi);
//
//    const double rootonehalf = sqrt(0.5);

    const int num = x.n_rows;

    double sumX=0.0;

    double sumX2=0.0;

    for(int i=0; i<num; i++)

    {

                    sumX += x(i);

                    sumX2 += x(i)*x(i);

    }
    double dev = sqrt(sumX2/(num-1));
    
    if(dev < 1e-100 * original_dev and dev < 1e-100) // deviation very small, then no error basically, return a big negative number
    {
        const double entropy = -1e100*log(1e100);
        cout << "Extremely small error deviation, set entropy = " << entropy 
             << ", dev = " << dev << ", original dev = " << original_dev
             << endl;
        return -entropy;
    }
    const double inv_dev = 1.0/dev;
    // Alternatively, use inv_dev1
    //const double dev = sqrt((trans(x)*x)(0,0)/double(num-1));
    //const double inv_dev1 = 1.0/dev;
//    cout << "k1 = " << k1 
//         << ", k2a = " << k2a 
//         << ", k2b = " << k2b 
//           
//         << ", dev = " << dev
//         << endl;
//
//    cout << "Calculating entropy for " << x.n_rows
//
//         << " elements, sample mean = " << sumX/num
//
//         << ", sample dev  = " << sqrt(sumX2/(num-1)) 
//        // << " (alt " << dev << " )"
//         << endl;
//
////    for(int i=0; i<num; i++)
//
////        cout << " " << x(i);
//
//    cout << endl;

    cout << "normalize factor log = " << log(inv_dev)
	     << ", inv_dev " << inv_dev
	     << ", Hnv = " << Hnu
	     << std::endl
	     ;
    if(0 == choice)

    {

        double Xexp = 0.0D;

        double Xabs = 0.0D;

        for(int i=0; i < num; i++)

        {

            const double sx = x(i)*inv_dev;

            Xexp += sx*exp(-0.5*sx*sx);

            Xabs += abs(sx);

        }

        Xexp /= num;

        Xabs /= num;

        const double XabsMinus = Xabs - root2overpi;

        cout << "choice 0, Xexp = " << Xexp << ", Xabs = " << Xabs

             << ", XabsMinus^2 = " << XabsMinus*XabsMinus << endl;
        return Hnu-(k1*Xexp*Xexp + k2a*XabsMinus*XabsMinus) - log(inv_dev);

    }

    else

    {

        double Xexp = 0.0D;

        double Xexp2= 0.0D;

        for(int i=0; i < num; i++)

        {

            const double sx = x(i)*inv_dev;

            const double tmp = exp(-0.5*sx*sx);

            Xexp += sx*tmp;

            Xexp2+= tmp;

        }

        Xexp /= num;

        Xexp2/= num;

        const double XexpMinus = Xexp2 - rootonehalf;

        cout << "choice 1, Xexp = " << Xexp << ", Xexp2 = " << Xexp2
             << ", XexpMinus^2 = " << XexpMinus*XexpMinus << endl;

        return Hnu-(k1*Xexp*Xexp + k2b*XexpMinus*XexpMinus) - log(inv_dev);

    }

}//end method calculate_entropy

//constructor
AdaptiveLassoEntropyScoringFunction::AdaptiveLassoEntropyScoringFunction(
    datastructures::BayesianNetwork& network
  , int which_score_
  , std::string data_file_name_
  , double lambda_
  , scoring::Constraints *constraints
  , bool is_adaptive_
  , bool enableDeCamposPruning
  )
  : which_score(which_score_)
  , variableCount(network.size())
  , num_rows(0) // will get correct value at the end of constructor
  , lambda(lambda_)
  , lambda2(2)
  , is_adaptive(is_adaptive_)
{
    this->network = network;
    this->constraints = constraints;
    this->enableDeCamposPruning = enableDeCamposPruning;
    beta_initialized_ = false;
    data_file_name = data_file_name_;
    // read data
    //arma::mat raw_data0;
    mlpack::data::Load(data_file_name, raw_data, true, false); // fatal=true, transpose=true
    //raw_data=trans(raw_data0);
    cout << "Dimension of raw_data, rows " << raw_data.n_rows
         << ", cols " << raw_data.n_cols
	 << ", lambda = " << lambda
         << endl;
    norm_data = raw_data;
    const_cast<int&>(variableCount) = raw_data.n_cols;
    const_cast<int&>(num_rows) = raw_data.n_rows;
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
        double dev_x2= dev_x; //is_adaptive ?dev_x : num_rows*dev_x; //sqrt(arma::dot(x,x)*num_rows);
        cout << "Raw data for variable #" << i 
             << ", mean " << mean_x 
             << ", dev " << dev_x 
             << ", dev2 "<< dev_x2 << ", dev2/dev = " << dev_x2/dev_x
             << ", sqrtN = " << sqrt(num_rows)
             << ", num_rows " << num_rows
             << ", variableCount " << variableCount
             << ", is_adaptive " << is_adaptive
             << endl;
        //normalize data
        for(int j=0; j<num_rows;j++)
        {
            raw_data(j,i)  = raw_data(j,i) - mean_x ;
            norm_data(j,i) = raw_data(j,i) / dev_x2 ; // normalize by dev_x2 = sqrt(num_rows)*dev_x, to avoid adjusting lambda with n_rows
        }
        raw_data_mean(i) = mean_x; 
        raw_data_dev(i) = dev_x;
        raw_data_entropy(i) = calculate_entropy(x, dev_x); // Calculate the independent entropy, use 0.0 as initial reference
        //raw_data_entropy(i) = calculate_entropy(norm_data.col(i), 1); // Calculate the independent entropy, use 0.0 as initial reference

        double lasso_score = 0;
        for(int j=0; j<num_rows;j++)
            lasso_score += x(j)*x(j);
	lasso_score /= dev_x*dev_x; // Newly added
	raw_data_lasso(i) = lasso_score;
	raw_data_bic(i) = num_rows*log(lasso_score/num_rows);// + log(num_rows); // k=1 for a single variance
    }
    // Now calculate the MLE beta matrix
    beta_MLE_mat=arma::mat(variableCount, variableCount);
    beta_MLE_mat.fill(0.0);
    arma::vec beta;
    arma::uvec parent_vec = arma::uvec(variableCount);
    int num_parents = 0;
    const double orig_lambda = lambda;
    const_cast<double&>(lambda) = 0.0D; // temporarily set lambda=0;

    printf("Printing beta_MLE_mat, lambda1=%5.2f, lambda2=%5.2f, orig_lambda=%5.2f, logn=%8.4f\n", lambda, lambda2, orig_lambda,log(raw_data.n_rows));
    printf("%6c", ' ');
    for(int i=0; i<variableCount; i++)
        printf(" var %02d", i+1);
    printf("\n");
    for(int i=0;i < variableCount; i++)
    {
        printf("Row %2d", i+1);
        for(int j=0; j<variableCount; j++)
              printf("%7.3f",beta_MLE_mat(i,j));
        printf("\n");
    }

    VARSET_NEW(parents, variableCount);
    VARSET_SET_ALL(parents, variableCount);
    for(int v=0; v < variableCount; v++)
    {
        VARSET_CLEAR(parents, v);
        float the_score = calculateScoreAndBeta(v, parents, false, beta, parent_vec, num_parents);
        
        // map beta back 
        cout << "BetaMLE, variable " << v << ", score " << the_score;
        for(int i=0; i<num_parents; i++)
        {
            beta_MLE_mat(v, parent_vec(i)) = beta(i);
            cout << ", (" << parent_vec(i) << ", " << beta_MLE_mat(v,parent_vec(i)) << ")";
        }
        cout << endl;
        VARSET_SET(parents, v);
    }
    beta_initialized_ = true;

    //const_cast<double&>(lambda) = is_adaptive ? 2.2*log(raw_data.n_rows) :orig_lambda*log(raw_data.n_rows);
    const_cast<double&>(lambda) = orig_lambda*log(raw_data.n_rows);
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

    printf("Printing beta_MLE_mat, lambda1=%5.2f, lambda2=%5.2f\n", lambda, lambda2);
    printf("%6c", ' ');
    for(int i=0; i<variableCount; i++)
        printf(" var %02d", i+1);
    printf("\n");
    for(int i=0;i < variableCount; i++)
    {
        printf("Row %2d", i+1);
        for(int j=0; j<variableCount; j++)
              printf("%7.3f",beta_MLE_mat(i,j));
        printf("\n");
    }
}// constructor AdaptiveLassoEntropyScoringFunction


float AdaptiveLassoEntropyScoringFunction::calculateScore(int variable, varset parents, FloatMap &cache)
{
    arma::vec beta;
    arma::uvec parent_vec = arma::uvec(variableCount);
    parent_vec.fill(0);
    int num_parents = 0;
    float the_score = calculateScoreAndBeta(variable, parents, is_adaptive, beta, parent_vec, num_parents);
    // check whether some beta's are zero
    //cout << "Clearing zero beta's bits for variable " << variable;
    int k=0;
    const double bic_threshold = 0; //4==which_score ? 2.0;
    varset orig_parents = parents;
    for(int i=0; i < num_parents; i++)
    {
        const int parent_index = parent_vec(i);
        varset thin_parents = orig_parents;
        VARSET_CLEAR(thin_parents, parent_index);
        FloatMap::const_iterator iter = cache.find(thin_parents);
        if(iter == cache.end())
        {
            // This is where we can prune by recording unrelated parent_index
            cout << "AdaptiveLassoBIC not accepting spurious parent, variable index " << variable
                 << ", spurious parent set " << varsetToString(thin_parents) 
                 << ", parent_index " << parent_index
                 << std::endl;
            return -the_score;
        }
        else if(iter->second + bic_threshold  >= -the_score )
        {

          cout << "AdaptiveLassoBIC not accepting spurious parent, variable index " << variable
               << ", good thin parent set " << varsetToString(thin_parents) 
               << ", the spurious parent_index " << parent_index
               << ", good thin parents score " << iter->second
               << ", the_score " << -the_score
               << std::endl
               ;
          return -the_score;
        }
    }

    cout << "Actual parameters k = " << k
	 << ", num_parents " << num_parents
	 << ", orig parents " << std::hex << orig_parents
	 << ", new parents " << parents
	 << std::dec
	 << endl;
    //cout << endl;
    //FloatMap::const_iterator iter = cache.find(parents);
    //if( iter == cache.end() and 4 != which_score )
      cache[parents] = -the_score;
    //else if( iter == cache.end() and 4 == which_score ) // BIC score , need check whether it contains spurious parent
    // {
    //  for( int i=0; i < num_parents; i++)
    //  {
    //	
    //  }
    //}
//    else
//    {
//        cout << "Variable # " << variable << "already has parents " << std::hex << parents << std::dec
//             << " with score  " << iter->second
//             << ", ignored new value " << -the_score
//             << endl;
//    }
    return -the_score;
}
float AdaptiveLassoEntropyScoringFunction::calculateScoreAndBeta(int variable, const varset & parents, const bool adaptive, arma::vec & beta, arma::uvec & parent_vec, int & num_parents)
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

    arma::rowvec Y=trans(raw_data.col(variable));
    const double dev_y = raw_data_dev(variable);
    //arma::rowvec Y=trans(norm_data.col(variable));
    //const double dev_y = 1.0; //raw_data_dev(variable);

//    cout << "num_parents = " 
//	 << num_parents 
//	 << ", which_score = " 
//	 << which_score
//	 << endl;
//
    if(0 == num_parents )
    {
        // Return 0.0 as there is no entropy reduction
        double entropy1 = raw_data_entropy(variable); //calculate_entropy(Y, raw_data_entropy(variable), raw_data_dev(variable));   //first approach
        // We calculate entropy2 just for comparison purpose
        double entropy2 = raw_data_entropy(variable); //calculate_entropy(Y, raw_data_entropy(variable), raw_data_dev(variable), 1); //second approach

        double lasso_score = 0;
        for(int i=0; i<num_rows;i++)
            lasso_score += Y(i)*Y(i);
	/*
        cout << "Calculated independent entropy, storing OLS score, variable # " << variable 
             << ", parents " << std::hex << parents << std::dec
             << ", dev " << sqrt(var(Y))
             << ", entropy1 = " << entropy1
             << ", entropy2 = " << entropy2 
             << ", lasso_score = " << lasso_score
             << ", rows = " << num_rows
             << ", cols = " << variableCount
             << ", which_score = " << which_score
	     << ", Y.n_rows = " << Y.n_rows
             << endl;
	*/
        double the_score = entropy1;
        switch(which_score)
        {
            case 0:
                the_score = raw_data_lasso(variable);
                break;
            case 2:
                the_score = entropy2;
                break;
	    case 3:
                the_score = 0.0;
		break;
	    case 4:
		the_score = raw_data_bic(variable);
		break;
            case 1:
            default:
                the_score = entropy1;
        }
        beta = arma::vec(variableCount);
        beta.fill(0.0);
        return the_score;
    }

    arma::mat X=norm_data.cols(parent_vec.head(num_parents));
    // adaptive here
    if(adaptive and beta_initialized_)
    {
        for(int i=0; i<num_parents;i++)
        {
            //betaMLE(i) = beta_MLE_mat(variable, parent_vec(i));
            // The following 
            double coef = beta_MLE_mat(variable, parent_vec(i));
            for(int k=0; k<X.n_rows; k++)
                X(k,i) *= coef;
        }
    }

    double varY = sqrt(var(Y));
    double varX0= sqrt(var(X.col(num_parents - 1)));
    // beta initialized
    if(beta_initialized_)
    {
      regression::LARS lasso1(true, lambda);
      lasso1.Train(X,Y,beta,false);// use true if X is column-major, and false if X is row-major
    }
    else // not initialized
    {
      regression::LARS lasso1(true, 0.0, lambda2);
      lasso1.Train(X,Y,beta,false);// use true if X is column-major, and false if X is row-major
    }
    
//    cout<<"beta.n_rows = "<< beta.n_rows
//        << ", num_parents = " << num_parents
//        << ", X.n_rows = " << X.n_rows
//        << ", X.n_cols = " << X.n_cols
//        << ", devY " << varY
//        << ", devX0  " << varX0
//        << ", MLE beta " << beta_MLE_mat(variable, parent_vec(num_parents-1))
//        <<endl;
//    arma::vec beta_full(variableCount);
//    for(int i=0; i < num_parents; i++)
//    {
//        beta_full
//    }
//    cout << "Before running Yhat, X dimension (" << X.n_rows
//	 << ", " << X.n_cols 
//	 << "), beta dimension(" << beta.n_rows
//	 << ", " << beta.n_cols 
//	 << "), Y dimension (" << Y.n_rows 
//	 << ", " << Y.n_cols << ")"
//	 << std::endl;
    arma::vec Yhat = X * beta ; //raw_data_mean(variable);
    //Yhat.fill(0.0);
    /*
    cout << "After running Yhat, Yhat dimension (" << Yhat.n_rows
	 << ", " << Yhat.n_cols
	 << "), Y dimension (" << Y.n_rows
	 << ", " << Y.n_cols
	 << "), beta dim(" << beta.n_rows
	 << ", " << beta.n_cols 
	 << ")"
	 << std::endl;
    */

    arma::vec error = Yhat - trans(Y);
    switch(which_score)
    {
        case 0: // LASSO 
	case 3: // L0
	case 4: // BIC
        {
            double sum_abs_beta = 0;
            double sum_error_sq = 0;
	    const int num_beta = beta.n_rows > beta.n_cols ? beta.n_rows : beta.n_cols ;
	    const int num_err = error.n_rows > error.n_cols ? error.n_rows : error.n_cols ;
	    int k=0;
            for(int i=0; i < num_beta ; i++)
	    {
                sum_abs_beta += abs(beta(i));
		k += abs(beta(i)) > 1e-8;
	    }
            for(int i=0; i < num_err ; i++)
                sum_error_sq += error(i)*error(i);

            double lasso_score = sum_error_sq + lambda*log(num_err)*sum_abs_beta ;
            double bic_score   = sum_error_sq + lambda*log(num_err)*k; // This is actually L0
            double the_score = 0 == which_score 
		             ? lasso_score - raw_data_lasso(variable) 
		             : 3 == which_score
			     ? bic_score   - raw_data_lasso(variable)
		             : ( num_err*log(sum_error_sq/num_err) + lambda*log(num_err)*k ) - raw_data_bic(variable)
			     ;
	    cout << "Calculated " << (which_score ? (4==which_score?"bic" : "L0") : "lasso" )
		 << " score for variable " << variable
		 << ", the_score " << -the_score
		 << ", independent_bic_score " << raw_data_bic(variable)
		 << ", parents " << varsetToString(parents)
		 << ",  num_beta " << num_beta
		 << ", beta dim(" << beta.n_rows << ", " << beta.n_cols
		 << "), error dim(" << error.n_rows << ", " << error.n_cols
		 << "), Y.n_rows = " << Y.n_rows
		 << ", num_err = " << num_err
		 << ", k = " << k
		 << ", sum_abs_beta = " << sum_abs_beta
		 << endl
		 ; 
            return the_score;
        }
        case 2: 
            return calculate_entropy(error, raw_data_dev(variable), 1) - raw_data_entropy(variable);
        case 1:
        default:
            return calculate_entropy(error, raw_data_dev(variable)) - raw_data_entropy(variable);
        
    }
}
void AdaptiveLassoEntropyScoringFunction::post_processing_one_parent(int variable, varset & parents)
{
    arma::vec beta;
    arma::uvec parent_vec = arma::uvec(variableCount);
    parent_vec.fill(0);
    int num_parents = 0;
    float the_score = calculateScoreAndBeta(variable, parents, false, beta, parent_vec, num_parents);
    cout << "Post processing, variable " << (1+variable)
         << ", score " << the_score << ", " << num_parents << " coefficients"
         ;
    for(int j=0; j<num_parents; j++)
    {
//        if(abs(beta(j)) < 1e-200)
//            continue;
        int col = parent_vec(j);
        cout << ", (" << (1+col) << ", " << beta(j) 
             << ", " << beta_MLE_mat(variable, col) 
//             << ", " << raw_data_dev(col)
             << ")";
    }
    cout << endl;
}
void AdaptiveLassoEntropyScoringFunction::post_processing(std::vector<int> & total_ordering, std::vector<varset>& optimalParents)
{
    // run lasso for post processing
    const_cast<double&>(lambda) = 0.0; // run Gaussian OLS
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
