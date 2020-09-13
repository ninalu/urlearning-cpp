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



#include "lasso_entropy_scoring_function.h"
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/core.hpp>
#include <cmath>
#include <iostream>


using namespace std;
using namespace mlpack;

namespace scoring
{
double LassoEntropyScoringFunction::calculate_entropy(const arma::vec & x, const double original_dev, int choice)

{
    const double pi = M_PI;

    const double k1 = 36.0/(8*sqrt(3) - 9.0);

    const double k2a = 1.0/(2.0-6.0/pi);

    const double k2b = 24/(16.0*sqrt(3)-27.0);

    const double Hnu = 0.5*(1.0+log(2.0*pi));

    const double root2overpi = sqrt(2.0/pi);

    const double rootonehalf = sqrt(0.5);

    const int num = x.n_rows;

    double sumX=0.0;

    double sumX2=0.0;

    for(int i=0; i<num; i++)

    {

                    sumX += x(i);

                    sumX2 += x(i)*x(i);

    }
    double dev = sqrt(sumX2/(num-1));
    
    if(dev < 1e-16 * original_dev and dev < 1e-16) // deviation very small, then no error basically, return a big negative number
    {
        const double entropy = -1e16*log(1e16);
        cout << "Extremely small error deviation, set entropy = " << entropy 
             << ", dev = " << dev << ", original dev = " << original_dev
             << endl;
        return -entropy;
    }
    const double inv_dev = 1.0/dev;
    // Alternatively, use inv_dev1
    //const double dev = sqrt((trans(x)*x)(0,0)/double(num-1));
    //const double inv_dev1 = 1.0/dev;
    cout << "k1 = " << k1 
         << ", k2a = " << k2a 
         << ", k2b = " << k2b 
           
         << ", dev = " << dev
         << endl;

    cout << "Calculating entropy for " << x.n_rows

         << " elements, sample mean = " << sumX/num

         << ", sample dev  = " << sqrt(sumX2/(num-1)) 
        // << " (alt " << dev << " )"
         << endl;

//    for(int i=0; i<num; i++)

//        cout << " " << x(i);

    cout << endl;

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
LassoEntropyScoringFunction::LassoEntropyScoringFunction(
    datastructures::BayesianNetwork& network
  , int which_score_
  , std::string data_file_name_
  , double lambda_
  , scoring::Constraints *constraints
  , bool enableDeCamposPruning
  )
  : which_score(which_score_)
  , variableCount(network.size())
  , num_rows(0) // will get correct value at the end of constructor
  , lambda(lambda_)
{
    this->network = network;
    this->constraints = constraints;
    this->enableDeCamposPruning = enableDeCamposPruning;
    data_file_name = data_file_name_;
    // read data
    //arma::mat raw_data0;
    raw_data.load(data_file_name);
    //raw_data=trans(raw_data0);
    cout << "Dimension of raw_data, rows " << raw_data.n_rows
         << ", cols " << raw_data.n_cols
         << endl;
    norm_data = raw_data;
    const_cast<int&>(variableCount) = raw_data.n_cols;
    const_cast<int&>(num_rows) = raw_data.n_rows;
    raw_data_mean = arma::vec(variableCount);
    raw_data_dev = arma::vec(variableCount);
    raw_data_entropy = arma::vec(variableCount);
    
    for(int i=0; i < variableCount; i++)
    {
        arma::vec x = raw_data.col(i);
        double mean_x = mean(x);
        x -= mean_x;
        double dev_x = sqrt(var(x));
        double dev_x2= num_rows*dev_x; //sqrt(arma::dot(x,x)*num_rows);
        cout << "Raw data for variable #" << i 
             << ", mean " << mean_x 
             << ", dev " << dev_x 
             << ", dev2 "<< dev_x2 << ", dev2/dev = " << dev_x2/dev_x
             << ", sqrtN = " << sqrt(num_rows)
             << ", num_rows " << num_rows
             << ", variableCount " << variableCount
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
    }
    // we should normalize raw_data here, named normalized_data =
    //normalize(raw_data);,
}

float LassoEntropyScoringFunction::calculateScore(int variable, varset parents, FloatMap &cache)
{
    // check if this violates the constraints
    if (constraints != NULL && !constraints->satisfiesConstraints(variable, parents)) {
        invalidParents.insert(parents);
        return variableCount;// trying to return a big number here, but not too big to cause overflow
    }
    
    arma::uvec parent_vec(variableCount);
    //arma::mat X(num_rows,1);
    //X.fill(1.0);
    int num_parents = 0;
    for(int i=0; i<variableCount; i++)
    {
        if(i != variable and VARSET_GET(parents, i))
        {
            parent_vec(num_parents++) = i;
            //X.insert_cols(num_parents++, raw_data.col(i));
            //cout<<X[i]<<endl;
        }
    }
    arma::vec Y=raw_data.col(variable);
    const double dev_y = raw_data_dev(variable);
    if(0 == num_parents )
    {
        // Return 0.0 as there is no entropy reduction
        double entropy1 = 0.0; //raw_data_entropy(variable); //calculate_entropy(Y, raw_data_entropy(variable), raw_data_dev(variable));   //first approach
        // We calculate entropy2 just for comparison purpose
        double entropy2 = 0.0; //raw_data_entropy(variable); //calculate_entropy(Y, raw_data_entropy(variable), raw_data_dev(variable), 1); //second approach

        double lasso_score = 0;
        for(int i=0; i<Y.n_rows;i++)
            lasso_score += Y(i)*Y(i);
        cout << "Calculated independent entropy, storing OLS score, variable # " << variable 
             << ", parents " << std::hex << parents << std::dec
             << ", dev " << sqrt(var(Y))
             << ", entropy1 = " << entropy1
             << ", entropy2 = " << entropy2 
             << ", lasso_score = " << lasso_score
             << ", rows = " << num_rows
             << ", cols = " << variableCount
             << endl;
        double the_score = entropy1;
        switch(which_score)
        {
            case 0:
                the_score = lasso_score;
                break;
            case 2:
                the_score = entropy2;
                break;
            case 1:
            default:
                the_score = entropy1;
        }
        cache[parents] = -the_score;
        return -the_score;
    }
    arma::mat X=norm_data.cols(parent_vec.head(num_parents));
    arma::vec beta;
    regression::LARS lasso1(true, lambda);
    lasso1.Train(X,Y,beta,false);// use true if X is column-major, and false if X is row-major
    cout<<"beta.n_rows = "<< beta.n_rows
        << ", num_parents = " << num_parents
        << ", X.n_rows = " << X.n_rows
        << ", X.n_cols = " << X.n_cols
        //<< ", beta = " << beta
        <<endl;
//    arma::vec beta_full(variableCount);
//    for(int i=0; i < num_parents; i++)
//    {
//        beta_full
//    }
    arma::vec Yhat= X*beta; //raw_data_mean(variable);
    //Yhat.fill(0.0);
    arma::vec error = Yhat- Y;
    const double error_dev = sqrt(var(error));
    double entropy1 = calculate_entropy(error, raw_data_dev(variable)) - raw_data_entropy(variable);   //first approach
    // We calculate entropy2 just for comparison purpose
    double entropy2 = calculate_entropy(error, raw_data_dev(variable), 1) - raw_data_entropy(variable); //second approach
    double sum_abs_beta = 0;
    double sum_error_sq = 0;
    for(int i=0; i<beta.n_rows;i++)
        sum_abs_beta += abs(beta(i));
    for(int i=0; i<error.n_rows;i++)
        sum_error_sq += error(i)*error(i);
    
    //double lasso_score = (trans(error)*error)(0,0) + lambda*sum_abs_beta; //ideally use this one, reduce one loop above
    double lasso_score = sum_error_sq + lambda*sum_abs_beta;
    cout << "Calculated dependent entropy, storing OLS score, variable # " << variable
         << ", parents " << std::hex << parents << std::dec
         << ", entropy1 = " << entropy1
         << ", entropy2 = " << entropy2
         << ", lambda " << lambda
         << ", lasso_score = " << lasso_score
         << ", rows = " << num_rows
         << ", cols = " << variableCount
         //<< ", beta " << beta(0) << ", " << beta(1) << ", " << beta(2) << ", " << beta(3) << ", " << beta(4)
         << endl;
    // if very few columns, print beta in full scale, i.e., those non-parent columns will  have coefficient 0
    if(variableCount < 10)
    { 
        int j=0;
        cout << "Beta vector: ";
        for(int i=0; i < variableCount; i++)
        {
            if(i==parent_vec(j))
            {
                cout << " " << beta(j)/raw_data_dev(j);
                j++;
            }
            else
                cout << "  0.0";
        }
        cout << endl;
    }
    double the_score = entropy1;
    switch(which_score)
    {
        case 0:
            the_score = lasso_score;
            break;
        case 2:
            the_score = entropy2;
            break;
        case 1:
        default:
            the_score = entropy1;
    }
    // check whether some beta's are zero
    cout << "Clearing zero beta's bits,";
    for(int i=0; i < num_parents; i++)
    {
        if(abs(beta(i)) < 0.01)
        {
            const int parent_index = parent_vec(i);
            VARSET_CLEAR(parents, parent_index);
            cout << " " << parent_index ;
        }
    }
    cout << endl;
    FloatMap::const_iterator iter = cache.find(parents);
    if( iter == cache.end())
      cache[parents] = -the_score;
    else 
    {
        cout << "parents " << std::hex << parents << std::dec
             << " already exists with score  " << iter->second
             << ", ignored new value " << -the_score
             << endl;
    }
    return -the_score;
}

}//end namespace scoring
