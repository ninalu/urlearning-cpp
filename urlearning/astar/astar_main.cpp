/* 
 * File:   main.cpp
 * Author: malone
 *
 * Created on August 6, 2012, 9:05 PM
 */

#include <string>
#include <iostream>
#include <ostream>
#include <vector>
#include <cstdlib>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "urlearning/base/node.h"
#include "urlearning/base/typedefs.h"
#include "urlearning/score_cache/score_cache.h"

#include "urlearning/priority_queue/priority_queue.h"
#include "urlearning/fileio/hugin_structure_writer.h"

#include "urlearning/score_cache/best_score_calculator.h"
#include "urlearning/score_cache/best_score_creator.h"

#include "urlearning/heuristic/heuristic.h"
#include "urlearning/heuristic/heuristic_creator.h"
#include "urlearning/base/skeleton.hpp"

#include "urlearning/scoring_function/bdeu_scoring_function.h"
#include "urlearning/scoring_function/bic_scoring_function.h"
#include "urlearning/scoring_function/fnml_scoring_function.h"
#include "urlearning/scoring_function/lasso_entropy_scoring_function.h"
#include "urlearning/scoring_function/adaptive_lasso_entropy.h"
//#define RUN_GOBNILP

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

void getFileCreationTime(char *path) {
    struct stat attr;
    stat(path, &attr);
    printf("%s, last modified time: %s", path, ctime(&attr.st_mtime));
}

namespace po = boost::program_options;


/**
 * A timer to keep track of how long the algorithm has been running.
 */
boost::asio::io_service io;

/**
 * A variable to check if the user-specified time limit has expired.
 */
bool outOfTime;
/**
 * The file containing the data.
 */
std::string inputFile;
std::string sf = "adaptive"; // scoring function
/**
 * The path to the score cache file.
 */
std::string scoreFile;
double lambda = 0.5; // for lasso 
bool adaptive = true;
int which = 2;
/**
 * The data structure to use to calculate best parent set scores.
 * "tree", "list", "bitwise"
 */
std::string bestScoreCalculator;

/**
 * The type of heuristic to use.
 */
std::string heuristicType;

/**
 * The argument for creating the pattern database.
 */
std::string heuristicArgument;

/**
 * The file to write the learned network.
 */
std::string netFile;
std::ofstream network_file;

/**
 * The ancestor-only variables for the search.
 * See UAI '14.
 */
std::string ancestorsArgument;       // csv list
varset ancestors;

/**
 * The variables for this search.
 */
std::string sccArgument;      //csv list
varset scc;

/**
 * The number of nodes expanded during the search.
 */
int nodesExpanded;

/**
 * the skeleton file, Ni added
 */
std::string skeletonFile;

/**
 * the scoring function to use in post-processing
 */
std::string scoring_function;

/**
 * The maximum running time for the algorithm.
 */
int runningTime;

/**
 * Handler when out of time.
 */
void timeout(const boost::system::error_code& /*e*/) {
    printf("Out of time\n");
    outOfTime = true;
}

std::vector<varset> reconstructSolution(Node *goal, std::vector<bestscorecalculators::BestScoreCalculator*> &spgs, NodeMap &closedList, const varset & scc, std::vector<int> & total_ordering) {
    std::vector<varset> optimalParents;
    for (int i = 0; i < spgs.size(); i++) {
        optimalParents.push_back(VARSET(spgs.size()));
    }

    VARSET_COPY(goal->getSubnetwork(), remainingVariables);
    Node *current = goal;
    float score = 0;
    int count = cardinality(scc);   //Ni added
    printf("reconstructSolution, before loop, parents %x, count %d\n", (unsigned int) remainingVariables, count);
    for (int i = 0; i < count; i++) {
        int leaf = current->getLeaf();
        total_ordering[count-1-i] = leaf;
        score += spgs[leaf]->getScore(remainingVariables);
        varset parents = spgs[leaf]->getParents();
        optimalParents[count-1-i] = parents;
	float the_score = spgs[leaf]->getScore(parents);

        printf("reconstructSolution, iteration %d, leaf %d, the_score %f, score %f, parents %x\n", i, leaf, the_score, score,(unsigned int)parents);
        VARSET_CLEAR(remainingVariables, leaf);

        current = closedList[remainingVariables];
    }

    return optimalParents;
}

int post_astar_process_count = 0;
void post_astar_processing(
  std::vector<bestscorecalculators::BestScoreCalculator*> & spgs
, int variableCount
, std::vector<int> & total_ordering
, std::vector<varset> & optimalParents
)
{
	post_astar_process_count++;
        float final_score = 0.0;
        for(int i=0; i < variableCount; i++)
        {
                int var_idx = total_ordering[i];
                final_score += spgs[var_idx]->getScore(optimalParents[i]);
        }
    printf("Finished setting parents for network, post_astar_process_count %d, count %d, total ordering: "
          , post_astar_process_count, variableCount);
    for(int i=0; i < variableCount; i++)
    {
        printf("%d ", total_ordering[i]);
    }
    printf("\n");

    // Ni added
    network_file << "NumVars " << variableCount << std::endl;
    for(int v=0; v < variableCount; v++)
    {
        const int var_idx = total_ordering[v];
        const varset & parents = optimalParents[v];
        printf("Variable # %d, parent bit set %lx, parents are (index starts from 1) ", var_idx + 1, parents);
        network_file << "Var " << total_ordering[v] + 1
                             << ", parents"
                                         ;

        for(int i=0; i < variableCount; i++)
        {
            if(VARSET_GET(parents,i))
            {
                printf("  %d", i+1);
                network_file << ", " << (i+1);
            }
        }
        printf("\n");
        network_file << std::endl;
    }
}

//Ni added
void run_astar_on_one_scc( int variableCount
                         , std::vector<bestscorecalculators::BestScoreCalculator*> & spgs
                         , heuristics::Heuristic* heuristic 
                         , scoring::ScoreCache & cache
                         , const varset & scc        // the original scc from Matlone
                         , const varset & ancestors  // ancestors
                         , const varset & the_scc    // our scc from skeleton
                         , int scc_idx
                         , datastructures::Skeleton & skeleton
                         )
{
    
    NodeMap generatedNodes;
    init_map(generatedNodes);

    PriorityQueue openList;
    // find one bit = 1, ideally from beginning
    int first_set_bit = VARSET_FIND_NEXT_SET(the_scc, 0);
    
    byte leaf(first_set_bit);   //original leaf =0, Ni modified
    Node *root = new Node(0.0f, 0.0f, ancestors, leaf);
    openList.push(root);
    
#ifdef DEBUG
    printf("The start is '%s'\n", varsetToString(ancestors).c_str());
    
    bool c;
    float lb = heuristic->h(ancestors, c);
    
    printf("The lb is %.2f\n", lb);
#endif

    Node *goal = NULL;
    VARSET_NEW(allVariables, variableCount);
    VARSET_SET_VALUE(allVariables, ancestors);
    allVariables = VARSET_OR(allVariables, the_scc);
    printf("allVariables equals the_scc = %d, scc#%d\n", allVariables == the_scc, scc_idx);
    printf("allVariables = %s\n", varsetToString(allVariables).c_str());
    printf("the_scc      = %s\n", varsetToString(the_scc).c_str());
    printf("original_scc = %s\n", varsetToString(scc).c_str());
#ifdef DEBUG
    printf("The goal is '%s'\n", varsetToString(allVariables).c_str());
#endif

    float upperBound = std::numeric_limits<float>::max();
    bool complete = false;

    printf("Beginning search\n");
    nodesExpanded = 0;
    VARSET_NEW(zero_varset, variableCount);
    while (openList.size() > 0 && !outOfTime) {
        //printf("top of while\n");
        Node *u = openList.pop();
        //printf("Expanding: '%s', g: '%.2f'\n", varsetToString(u->getSubnetwork()).c_str(), u->getG());
        nodesExpanded++;

        //        if (nodesExpanded % 100 == 0) {
        //            std::cout << ".";
        //            std::cout.flush();
        //        }

        varset variables = u->getSubnetwork();

        // check if it is the goal
        if (variables == allVariables) {
            printf("Found goal node, total cost = %f\n, leaf = %d\n", u->getG(), u->getLeaf());
            goal = u;
            break;
        }

        // TODO: this is mostly broken
        if (u->getF() > upperBound) {
            printf("Stop loop due to too high cost, f = %f, g = %f\n", u->getF(), u->getG());
            break;
        }

        // note that it is in the closed list
        u->setPqPos(-2);

        // expand
        // Ni added, more efficiently, calculate set of neighbors from variables, then exclude variables, 
        // get the candidate list, to be improved later
        for (byte leaf = 0; leaf < variableCount; leaf++) {
            // make sure this variable was not already present
            if (VARSET_GET(variables, leaf)) continue;
            
            // also make sure this is one of the variables we care about
            if (!VARSET_GET(the_scc, leaf)) continue;
            //Ni added the following if-statement
            if(skeleton.good() and not VARSET_EQUAL(variables, zero_varset))//excluding the empty set so that no skeleton checking at startup
            {
                const varset & neighbors = skeleton.get_neighbors(leaf);
                if(VARSET_EQUAL(VARSET_AND(variables, neighbors), zero_varset))
                {
                    //printf("Skipping leaf %d because it is not in neighbors %16lx\n", leaf, neighbors);
                    continue;
                }
            }
            // get the new variable set
            VARSET_COPY(variables, newVariables);
            VARSET_SET(newVariables, leaf);


            Node *succ = generatedNodes[newVariables];

            // check if this is the first time we have generated this node
            if (succ == NULL) {
                // get the cost along this path
                
                //printf("About to check for using leaf '%d', newVariables: '%s'\n", leaf, varsetToString(newVariables).c_str());
                float leaf_score = spgs[leaf]->getScore(newVariables);
                float g = u->getG() + leaf_score; //spgs[leaf]->getScore(newVariables);
                //printf("I have g\n");
                // calculate the heuristic estimate
                complete = false;
                float h = heuristic->h(newVariables, complete);
                //printf("I have h: %.2f\n", h);

#ifdef RUN_GOBNILP
                std::string filename = "exclude_" + TO_STRING(newVariables) + ".pss";
                time_t t;
                double seconds;

                t = time(NULL);
                int count = cache.writeExclude(filename, newVariables);
                seconds = difftime(time(NULL), t);

                // check whether to execute or not
                if (count < 10000 && variableCount > 25) {

                    // now, call gobnilp on the subproblem
                    FILE *fpipe;
                    char buffer[1048576];
                    char cmd[4096];
                    int i;

                    t = time(NULL);
                    snprintf(cmd, sizeof (cmd), "gobnilp %s", filename.c_str());

                    if (0 == (fpipe = (FILE*) popen(cmd, "r"))) {
                        perror("popen() failed.");
                        exit(1);
                    }

                    while ((i = fread(buffer, sizeof (char), sizeof buffer, fpipe))) {


                        /* We found some solutions, so split them up and add the cutting planes. */

                        // make sure to clean up the buffer so it behaves as a string
                        buffer[i] = 0;

                        //printf("buffer: '%s'\n", buffer);

                        char *pch;
                        pch = strtok(buffer, "\n");

                        bool optimal = false;

                        while (pch != NULL) {

                            // check if this is a "Dual Bound         : -4.62524687000000e+02" line

                            // I believe this is the only line in the output which starts with a 'D'
                            // this is really brittle, though
                            if (pch[0] == 'D') {
                                std::vector<std::string> tokens;
                                boost::split(tokens, pch, boost::is_any_of(" "), boost::token_compress_on);
                                float score = -1 * atof(tokens[3].c_str());
                                printf("Node: %s, g: %f, (%s) h(gobnilp): %f, f(gobnilp): %f, h(pd): %f, f(pd): %f\n",
                                        varsetToString(newVariables).c_str(), g, (optimal ? "optimal" : "bound"), score, (score + g), h, (g + h));
                                h = score;
                            }

                            // also, check if this line contains the word "optimal"
                            // if so, then this is actually the globally optimal solution
                            if (strstr(pch, "optimal") != NULL) {
                                optimal = true;
                            }

                            pch = strtok(NULL, "\n");
                        }
                    }
                    pclose(fpipe);

                    remove(filename.c_str());

                    seconds = difftime(time(NULL), t);
                    printf("Time to solve and parse ip: %f(s)\n", seconds);
                }

#endif

#ifdef CHECK_COMPLETE
                if (complete) {
                    float score = g + h;
                    if (score < upperBound) {
                        upperBound = score;
                        printf("new upperBound: %f, nodes expanded: %d, open list size: %d\n", upperBound, nodesExpanded, openList.size());
                    }
                    continue;
                }
#endif

                // update all the values
                succ = new Node(g, h, newVariables, leaf);
                
                //printf("I have created succ\n");

                // add it to the open list
                openList.push(succ);
                
                //printf("pushed succ\n");

                // and to the list of generated nodes
                generatedNodes[newVariables] = succ;
                //printf("added succ to generatedNodes\n");
                continue;
            }

            // assume the heuristic is consistent
            // so check if it was in the closed list
            if (succ->getPqPos() == -2) {
                continue;
            }
            // so we have generated a node in the open list
            // see if the new path is better
            float g = u->getG() + spgs[leaf]->getScore(variables);
            if (g < succ->getG()) {
                // the update the information
                succ->setLeaf(leaf);
                succ->setG(g);

                // and the priority queue
//                if (succ->getPqPos() == -2)
//                {
//                	succ->setPqPos(0);
//                	openList.push(succ);
//                }
//                else
                openList.update(succ);
            }
        }
    }

    printf("Nodes expanded: %d, open list size: %d\n", nodesExpanded, openList.size());
    
    heuristic->printStatistics();

    if (goal != NULL) {
        //printf("Found solution: %f\n", goal->getF());
        printf("Found solution: %f, scc # %d, netFile.length %lu\n", goal->getG(), scc_idx, netFile.length());

        if (netFile.length() > 0) {
	    network_file.open(netFile.c_str(), std::ios_base::trunc);
            printf("Opened network file %s\n", netFile.c_str());
            datastructures::BayesianNetwork *network = cache.getNetwork();
            network->fixCardinalities();
            std::vector<int> total_ordering(variableCount);
            std::vector<varset> optimalParents = reconstructSolution(goal, spgs, generatedNodes, the_scc, total_ordering);
            printf("Finished reconstructSolution\n");
            network->setParents(optimalParents);
            printf("Setting uniform probability\n");
            //network->setUniformProbabilities();
            // post processing
            printf("Start post-processing\n");
            scoring::ScoringFunction *scoringFunction;
            if (sf.compare("lasso") == 0 or sf.compare("entropy") == 0){
                printf("Creating Entropy/LassoFunction with input file %s\n", inputFile.c_str());
                scoringFunction = new scoring::LassoEntropyScoringFunction(*network, which, inputFile, lambda, NULL, false);
            } else if (sf.compare("adaptive_lasso") == 0 or sf.compare("adaptive") == 0){
                printf("Creating Adaptive Entropy/LassoFunction with input file %s\n", inputFile.c_str());
                scoringFunction = new scoring::AdaptiveLassoEntropyScoringFunction(*network, which, inputFile, lambda, NULL, adaptive, false);
            }
            // run post_processing
            scoringFunction->post_processing(total_ordering, optimalParents);
	    post_astar_processing(spgs,variableCount, total_ordering, optimalParents); // just print network file
	    network_file.close();

            printf("Finished setting parents for network, count %d, total ordering: ", variableCount);
            for(int i=0; i < variableCount; i++)
            {
                printf("%d ", total_ordering[i]);
            }
            printf("\n");
// Ni commented  out
//           fileio::HuginStructureWriter writer;
//           writer.write(network, netFile);
            // Ni added
            std::vector<uint64_t> vpar(variableCount, 0L);
            for(int v=0; v < variableCount; v++)
            {
                const varset & parents = optimalParents[v];
                vpar[total_ordering[v]] = parents;
                printf("Variable # %d, parent bit set %lx, parents are (index starts from 0) ", total_ordering[v], parents);
                for(int i=0; i < variableCount; i++)
                {
                    if(VARSET_GET(parents,i))
                        printf("  %d", i);
                }
                printf("\n");
            }
            std::string netFile2 = netFile + ".csv";
            std::ofstream network_file2(netFile2.c_str(), std::ios_base::trunc);
            for(int v=0; v<variableCount; v++)
            {
            	const varset parents = vpar[v];
            	char buf[256];
            	int length = 0;
            	for(int i=0; i<variableCount; i++)
            	{
            		length += snprintf(buf+length, 4, "%d,", (VARSET_GET(parents, i) ? 1 : 0));
            	}
            	buf[length-1]='\n';
            	buf[length] = '\0';
            	network_file2 << buf;
            }
            network_file2.close();
        }
    } else {
        Node *u = openList.pop();
        printf("No solution found.\n");
        if(u)
          printf("Lower bound: %f\n", u->getF());
    }
    
    for (auto pair = generatedNodes.begin(); pair != generatedNodes.end(); pair++) {
        delete (pair->second);
    }
    
}// run_astar_on_one_scc

void astar() {
    printf("URLearning, A*\n");
    printf("Dataset: '%s'\n", scoreFile.c_str());
    printf("Net file: '%s'\n", netFile.c_str());
    printf("Best score calculator: '%s'\n", bestScoreCalculator.c_str());
    printf("Heuristic type: '%s'\n", heuristicType.c_str());
    printf("Heuristic argument: '%s'\n", heuristicArgument.c_str());
    printf("Ancestors: '%s'\n", ancestorsArgument.c_str());
    printf("SCC: '%s'\n", sccArgument.c_str());

    boost::timer::auto_cpu_timer act;

    
    printf("Reading score cache.\n");
    scoring::ScoreCache cache;
    cache.read(scoreFile);
    printf("Done reading score cache. Verifying \n");
    int variableCount = cache.getVariableCount();
    printf("Variable count is %d\n",variableCount);
    
    const int num = variableCount;
    varset optimal_parents[num];
    //VARSET_NEW(vs0, num);
    //VARSET_NEW(vs1, num);
    //VARSET_NEW(vs2, num); VARSET_SET(vs2, 0);
    //VARSET_NEW(vs3, num); VARSET_SET(vs2, 0); VARSET_SET(vs2, 1);
    //VARSET_NEW(vs4, num); VARSET_SET(vs4, 2); VARSET_SET(vs4, 3);

    
    //optimal_parents[0] = vs0;
    //optimal_parents[1] = vs1;
    //optimal_parents[2] = vs2;
    //optimal_parents[3] = vs3;
    //optimal_parents[4] = vs4;
    for(int variable=0; variable<num; variable++)
    {
        //Ni added, print out score
        printf("Independent and dependent scores, variable # %d, ind_score %f, depend_score %f\n"
              , variable, cache.getScore(variable, 0)
              , cache.getScore(variable, optimal_parents[variable]) );
    }
    
    // parse the ancestor and scc variables
    VARSET_NEW(ancestors,variableCount);
    VARSET_NEW(scc,variableCount);
    
    // if no scc was specified, assume we just want to learn everything
    if (sccArgument == "") {
        VARSET_SET_ALL(scc, variableCount);
    } else {
        setFromCsv(scc, sccArgument);
        setFromCsv(ancestors, ancestorsArgument);
    }
    
    
    act.start();
    printf("Creating BestScore calculators.\n");
    std::vector<bestscorecalculators::BestScoreCalculator*> spgs = bestscorecalculators::create(bestScoreCalculator, cache);
    act.stop();
    act.report();

    act.start();
    printf("Creating heuristic.\n");
    heuristics::Heuristic *heuristic = heuristics::createWithAncestors(heuristicType, heuristicArgument, spgs, ancestors, scc);

#ifdef DEBUG
    heuristic->print();
#endif

    act.stop();
    act.report();
    
    datastructures::Skeleton skeleton; // Ni added
	if(skeletonFile.find(".arc") + 4 == skeletonFile.size())
		skeleton.read_arc_list_file(skeletonFile, variableCount);
	else
	    skeleton.read_matrix_file(skeletonFile); // Ni added
    if(not skeleton.good())
        skeleton.set_variable_count(variableCount);
    printf("Checking skeleton, good = %d\n", skeleton.good());
    
    const std::vector<varset> & scc_list = skeleton.get_scc();//Ni added
    const int num_scc = scc_list.size();    //Ni added
    printf("num of sccs = %d\n", num_scc);
    act.start();
    for(int i=0; i < num_scc; i++)  //Ni added
    {
        run_astar_on_one_scc(variableCount, spgs, heuristic, cache, scc, ancestors, scc_list[i], i, skeleton);
    }
    for (auto spg = spgs.begin(); spg != spgs.end(); spg++) {
        delete *spg;
    }
    delete heuristic;
    act.stop();
    act.report();
    io.stop();
}//astar

int main(int argc, char** argv) {
	
	getFileCreationTime(argv[0]); // print the binary executable creation time
	
    boost::timer::auto_cpu_timer act;
    srand(time(NULL));

    std::string description = std::string("Learn an optimal Bayesian network using A*.  Example usage: ") + argv[0] + " iris.pss";
    po::options_description desc(description);

    desc.add_options()
            ("scoreFile", po::value<std::string > (&scoreFile)->required(), "The file containing the local scores in pss format. First positional argument.")
            ("skeleton,k", po::value<std::string > (&skeletonFile), "The file containing the edges of a skeleton")
            ("scoring_function,f", po::value<std::string>(&sf), "The scoring function to use in post-processing")
            ("raw_inputFile,i", po::value<std::string>(&inputFile), "The raw data file to use in scoring and post-processing")
            ("lambda,l", po::value<double> (&lambda), "The lambda in Lasso.")
            ("adaptive", "Use adaptive Lasso")
            ("scoreType,w", po::value<int> (&which)->default_value(1), "which score, 0 for lasso, 1 for entropy1, 2 for entropy2")
            ("bestScore,b", po::value<std::string > (&bestScoreCalculator)->default_value("list"), bestscorecalculators::bestScoreCalculatorString.c_str())
            ("heuristic,e", po::value<std::string > (&heuristicType)->default_value("static"), heuristics::heuristicTypeString.c_str())
            ("argument,a", po::value<std::string > (&heuristicArgument)->default_value("2"), heuristics::heuristicArgumentString.c_str())
            ("pc_{i-1},p", po::value<std::string > (&ancestorsArgument)->default_value(""), "Variables which can only be used as ancestors. They will not be added in the search. CSV-list of variable indices. See UAI '14.")
            ("scc_i,s", po::value<std::string > (&sccArgument)->default_value(""), "Variables which will be added in the search. CSV-list of variable indices. Leave blank to add all variables. See UAI '14.")
            ("runningTime,r", po::value<int> (&runningTime)->default_value(0), "The maximum running time for the algorithm.  0 means no running time.")
            ("netFile,n", po::value<std::string > (&netFile)->default_value(""), "The file to which the learned network is written.  Leave blank to not create the file.")
            ("help,h", "Show this help message.")
            ;

    po::positional_options_description positionalOptions;
    positionalOptions.add("scoreFile", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc)
            .positional(positionalOptions).run(),
            vm);

    if (vm.count("help") || argc == 1) {
        std::cout << desc;
        return 0;
    }
    if(vm.count("adaptive"))
    {
        adaptive = true;
        std::cout << "Will use adaptive Lasso\n";
    }
    po::notify(vm);
    outOfTime = false;

    boost::to_lower(bestScoreCalculator);

    boost::asio::deadline_timer t(io);
    if (runningTime > 0) {
        printf("Maximum running time: %d\n", runningTime);
        t.expires_from_now(boost::posix_time::seconds(runningTime));
        t.async_wait(timeout);
        boost::thread workerThread(astar);
        io.run();
        workerThread.join();
    } else {
        astar();
    }

    return 0;
}

