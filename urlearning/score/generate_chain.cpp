/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <cstdlib>
#include <cstdio>
#include <random>
#include <iostream>
#include <armadillo>
#include <ctime>
#include <chrono>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

std::default_random_engine generator;

std::normal_distribution<double> distribution1(0,1);

std::chi_squared_distribution<double> distribution2(1.0);

std::student_t_distribution<double> distribution3(1.0);

std::uniform_real_distribution<double> distribution4(-1.0,1.0);

std::exponential_distribution<double> distribution5(1.0);

std::geometric_distribution<int> distribution6(0.5);


//double laplace_random_number = distribution5(generator) ;
//
//if(distribution4(generator) < 0.0 )
//
//   laplace_random_number = - laplace_random_number;

double get_random_number(int distribution)
{
    switch(distribution)
    {
        case 1: // Normal distribution
            return distribution1(generator);
        case 2: // chi-square
            return distribution2(generator);
        case 3: // student-t
            return distribution3(generator);
        case 4: // uniform
            return distribution4(generator);
        case 5: // exponential
            return distribution5(generator);
        case 6: // geometric
            return distribution6(generator);
        case 7:
            return distribution5(generator)*(distribution4(generator) < 0.0 ? -1 : 1);
        default:
            return 0.0;
            ;
    }
}

int get_distribution_code(const char* var_dist)
{
    if(strcmp(var_dist,"normal") == 0)
        return 1;
    else if(strcmp(var_dist,"chi-square") == 0)
        return 2;
    else if(strcmp(var_dist,"student-t") == 0)
        return 3;
    else if(strcmp(var_dist,"uniform") == 0)
        return 4;
    else if(strcmp(var_dist,"exponential") == 0)
        return 5;
    else if(strcmp(var_dist,"geometric") == 0)
        return 6;
    else if(strcmp(var_dist, "laplace")== 0)
        return 7;
    return 0;
}

int main(int argc, char** argv)

{
    //int64_t seed = time(NULL);
    //generator.seed(seed);
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    double signal_noise_ratio = 6.0;
    int num_vars = 2;
    int dependent_var_idx = 0;
    int distribution = 1; // 1 for normal distribution
    int noise_distribution = 1; 
    std::string outputFile;
    int N = 1000;//sample size, number of rows
    std::string description = std::string("Generate a csv file.  Example usage: ") + argv[0] + " ";
//    po::options_description desc(description);
//    desc.add_options()
//            ("signal,s", po::value<double > (&signal_noise_ratio), "signal noise ratio")
//            ("output,o", po::value<std::string > (&outputFile), "The output file. Second positional argument.")
//            ("num_independents,i", po::value<int>(&num_vars)->default_value(1), "The number of independent variables")
//            ("distribution,d", po::value<int>(&distribution)->default_value(1)
//              , (std::string("The probability distributions\n\t 1 for normal distribution\n2 for chi-square")
//             // + "\n\t 3 for student-t distribution"
//             // + "\n\t 4 for uniform distribution"
//             // + "\n\t 5 for exponential distribution"
//              + "\n\t 6 for geometric distribution").c_str()
//            )
//            ("noise_distribution,z", po::value<int>(&noise_distribution)->default_value(1), "noise distribution")
//            ("sample size, n", po::value<int>(&N)->default_value(1000), "number of rows")
//            ("help,h", "Show this help message.")
//            ;

//    po::positional_options_description positionalOptions;
//    positionalOptions.add("input", 1);
//    positionalOptions.add("output", 1);

//    po::variables_map vm;
//    po::store(po::parse_command_line(argc, argv, desc), vm);
//    po::notify(vm);
//    if (vm.count("help") || argc == 1) {
//        std::cout << desc;
//        return 0;
//    }
    
    // temporary solution due to program options not working
    if(argc < 4)
    {
        printf("Usage: %s num_vars distribution noise_distribution noise_strength\n", argv[0]);
        exit(1);
    }
    num_vars = atoi(argv[1]);
    distribution = get_distribution_code ( argv[2] );
    noise_distribution = get_distribution_code( argv[3] );
    
    double noise_strength = 1.0;
    if(argc > 4)
        noise_strength = atof(argv[4]);
 

       const int cols =  num_vars + 1;

       double data[N][cols];

       //arma::vec error(N);



       double sumX=0.0;

       double sumX2=0.0;
       
       //dependent_var_idx = int((get_random_number(4)+1) * cols + 1) % cols;
       arma::vec beta(num_vars+1);
       //beta(dependent_var_idx) = 0.0;
       std::vector<int> indices(cols);
       for(int j=0;j<cols;j++)
           indices[j] = j;
       
       //random_shuffle(indices.begin(), indices.end()); 
       
       for(int i=0; i < N; i++)
       {
           data[i][ indices[0] ] = get_random_number(distribution);
       }
       for(int j=1; j < cols; j++)
       {   
           const int curr = indices[j];
           const int prev = indices[j-1];
           beta[curr] = get_random_number(4) + 8.0;// range (6, 11) + signal_noise_ratio;
           if(abs(beta[curr]) < signal_noise_ratio)
               beta[curr] = abs(beta[curr])/beta[curr]*signal_noise_ratio;
           //double strength = int((get_random_number(4)) * 2000) % 50;

           for(int i=0; i<N; i++)
           {
                data[i][curr] = beta[curr] * data[i][prev] + noise_strength*get_random_number(noise_distribution);
           }
       }
       
       for(int j=1; j < cols; j++)
           printf("# %d -> %d, %12.4f\n", indices[j-1], indices[j], beta(indices[j]));
       printf("\n");
       for(int i=0; i<N; i++)
       {
           for(int j=0; j < cols; j++)
              printf("%12.4f%s ", data[i][j], j == cols-1 ? "" : ",");
           printf("\n");
       }

       return 0;
}

