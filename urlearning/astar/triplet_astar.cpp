/*
 * big_astar.cpp
 *
 *  Created on: Feb 24, 2019
 *      Author: nilu
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
#include "urlearning/base/bayesian_network.h"
#include "undirected_edges.hpp"
#include "urlearning/base/EdgeConstraints.h"

//#include "urlearning/scoring_function/bdeu_scoring_function.h"
//#include "urlearning/scoring_function/bic_scoring_function.h"
//#include "urlearning/scoring_function/fnml_scoring_function.h"
//#include "urlearning/scoring_function/lasso_entropy_scoring_function.h"
#include "urlearning/scoring_function/adaptive_lasso_entropy.h"
#include "neighbors.h"
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
std::ofstream network_file_csv;

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

/*
std::vector<varset> reconstructSolution( Node *goal
		                       , std::vector<bestscorecalculators::BestScoreCalculator*> &spgs
				       , NodeMap &closedList
				       , const varset & scc
				       , std::vector<int> & total_ordering
				       ) 
{
    std::vector<varset>  optimalParents;
    for (int i = 0; i < spgs.size(); i++) {
        optimalParents.push_back(VARSET(spgs.size()));
    }

    VARSET_COPY(goal->getSubnetwork(), remainingVariables);
    Node *current = goal;
    float score = 0;
    int count = cardinality(scc);   //Ni added
    printf("reconstructSolution, before loop, parents %x, count %d\n", (unsigned int) remainingVariables, count);
    for (int i = 0; i < count; i++) 
    {
        int leaf = current->getLeaf();
        total_ordering[count-1-i] = leaf;
        score += spgs[leaf]->getScore(remainingVariables);
        varset parents = spgs[leaf]->getParents();
        optimalParents[count-1-i] = parents;

        printf("reconstructSolution, iteration %d, leaf %d, score %f, parents %x\n", i, leaf, score,(unsigned int)parents);
        VARSET_CLEAR(remainingVariables, leaf);

        current = closedList[remainingVariables];
    }

    return optimalParents;
}
*/
void reconstructSolution( Node *goal
		        , std::vector<bestscorecalculators::BestScoreCalculator*> &spgs
			, NodeMap &closedList
			, const varset & scc
			, std::vector<int> & total_ordering
                        , std::vector<varset> & optimalParents
                        , std::vector<varset> & optimalChildren
			, std::vector<float> & optimalScores
			) 
{
    optimalParents.reserve(spgs.size());
    optimalChildren.reserve(spgs.size());
    optimalScores.reserve(spgs.size());
    for (size_t i = 0; i < spgs.size(); i++) 
    {
        optimalParents.push_back(VARSET(spgs.size()));
        optimalChildren.push_back(VARSET(spgs.size()));
	optimalScores.push_back(0.0);
    }
    const int variableCount = spgs.size();

    VARSET_COPY(goal->getSubnetwork(), remainingVariables);
    Node *current = goal;
    float score = 0;
    int count = cardinality(scc);   //Ni added
    printf("reconstructSolution, before loop, parents %x, count %d, variableCount %d\n", (unsigned int) remainingVariables, count, variableCount);
    std::vector<int> var2ordering(variableCount, -1);
    for (int i = 0; i < count; i++) 
    {
        int leaf = current->getLeaf();
        total_ordering[count-1-i] = leaf;
	var2ordering[leaf] = count-1-i;
	float the_score = spgs[leaf]->getScore(remainingVariables);
        score += the_score;
        varset parents = spgs[leaf]->getParents();
        optimalParents[count-1-i] = parents;
	optimalScores[count-1-i] = the_score;

        printf("reconstructSolution, iteration %d, leaf %d, score %f, optimal score_delta %f, parents %lx\n", i, leaf, score,the_score,(uint64_t)parents);
        VARSET_CLEAR(remainingVariables, leaf);

        current = closedList[remainingVariables];
    }
    for(int i=0; i<count; i++)
    {
	int var = total_ordering[i];
	varset parents = optimalParents[i];
	for(int j=0; j<variableCount; j++)
	{
	   if(VARSET_GET(parents, j) ) // variable j is a parent of var
	   {
		int jdx = var2ordering[j];
		VARSET_SET(optimalChildren[jdx], var);
	   }
	}
    }
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
        printf("Variable # %d, parent bit set %lx, parents are (index starts from 0) ", var_idx , parents);
        network_file << "Var " << total_ordering[v] 
                             << ", parents"
                                         ;

        for(int i=0; i < variableCount; i++)
        {
            if(VARSET_GET(parents,i))
            {
                printf("  %d", i);
                network_file << ", " << i;
            }
        }
        printf("\n");
        network_file << std::endl;
    }
}

//Ni added
void run_astar_on_one_scc( int variableCount
                         , std::vector<bestscorecalculators::BestScoreCalculator*> & spgs
                        // , heuristics::Heuristic* heuristic
                         , scoring::ScoreCache & cache
                         , const varset & scc        // the original scc from Matlone
                         , varset & ancestors  // ancestors
                         , varset & the_scc    // our scc from skeleton
			 , const datastructures::EdgeConstraints & constraints
                         , int scc_idx
                         , const datastructures::Skeleton & skeleton
			 , std::vector<varset> & optimal_parents
			 , std::vector<varset> & optimal_children
			 , const char* marker = ""
                         )
{

//    act.start();
    printf("Creating heuristic, scc_idx %d, heuristic_type %s, ancestors %lx, the_scc %lx.\n", scc_idx, heuristicType.c_str(), ancestors, the_scc);
    heuristics::Heuristic *heuristic = heuristics::createWithAncestors(heuristicType, heuristicArgument, spgs, ancestors, the_scc);

#ifdef DEBUG
    heuristic->print();
#endif

//    act.stop();
//    act.report();

    optimal_parents.clear();
    optimal_children.clear();
    optimal_parents.resize(variableCount);
    optimal_children.resize(variableCount);

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

    std::map<float, Node*> goals;
    float upperBound = std::numeric_limits<float>::max();
    bool complete = false;

    printf("Line %d, beginning search, scc_idx = %d\n", __LINE__, scc_idx);
    nodesExpanded = 0;
    while (openList.size() > 0 && !outOfTime) {
        //printf("top of while\n");
        Node *u = openList.pop();
        //printf("Line %d, expanding: '%s', g: %.2f, f=%f\n", __LINE__, varsetToString(u->getSubnetwork()).c_str(), u->getG(), u->getF());
        nodesExpanded++;

        //        if (nodesExpanded % 100 == 0) {
        //            std::cout << ".";
        //            std::cout.flush();
        //        }

        varset variables = u->getSubnetwork();

        // check if it is the goal
        if (variables == allVariables) 
	{
            goals.insert(std::pair<float, Node*>(u->getG(), u)); 
	    float best_score = goals.begin()->first;
	    Node* nu = openList.size() ? openList.peek() : NULL;
	    goal = goals.begin()->second;
            printf("Line %d, found goal node, total cost = %f, leaf = %d, num_goal_nodes %d, best score %f, openList length %d, best score in openList %f, goal==u : %d \n"
	    , __LINE__, goal->getG(), goal->getLeaf(), int(goals.size()), best_score, openList.size(), nu ? nu->getF() : 0.0, goal==u);
            goal = u;
	    /*
	    if(openList.size()==0)
	    {
		printf("Line %d, No more node in openList, breaking out of loop\n", __LINE__);
                break;
	    }
	    if(best_score < nu->getF() - 3.0)
	    {
		printf("Line %d, best score better than best estimate in openList, breaking out of loop\n", __LINE__);
		break;
	    }
	    printf("Line %d, found goal node, but will check further, best score %f, u->f=%f\n", __LINE__, best_score, u->getG());
	    continue;
	    */
	    break;
        }

        // TODO: this is mostly broken
        if (u->getF() > upperBound) {
            printf("Line %d, stop loop due to too high cost, f = %f, g = %f\n", __LINE__, u->getF(), u->getG());
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
	    /*
            if(skeleton.good() and not VARSET_EQUAL(variables, zero_varset))//excluding the empty set so that no skeleton checking at startup
            {
                const varset neighbors = skeleton.get_neighbors(leaf);
                if(VARSET_EQUAL(VARSET_AND(variables, neighbors), zero_varset))
                {
                    //printf("Skipping leaf %d because it is not in neighbors %16lx\n", leaf, neighbors);
                    continue;
                }
            }
	    */
            if(not constraints.satisfiesConstraints(leaf, variables, the_scc))// this leaf required parents are not there
            {
		continue;
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
//                printf("Line %d, leaf %d, parents %s, vars_so_far %s, leaf_score %f, g=%f, h=%f, g+h=%f \n"
//		, __LINE__, leaf, varsetToString(variables).c_str(), varsetToString(newVariables).c_str(), leaf_score, g, h, g+h);

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
		//Node* nu = openList.peek();

                //printf("pushed succ\n");

                // and to the list of generated nodes
                generatedNodes[newVariables] = succ;
                //printf("Line %d, added succ to generatedNodes, leaf %d, g=%f, h=%f, f=%f, next best score in list %f, variables %s, openList size %d\n"
		//, __LINE__, leaf, g, h, succ->getF(), varsetToString(newVariables).c_str(), nu->getF(), openList.size() );
                continue;
            }

            // assume the heuristic is consistent
            // so check if it was in the closed list
//            if (succ->getPqPos() == -2) {
//                continue;
//            }
            // so we have generated a node in the open list
            // see if the new path is better
            float g = u->getG() + spgs[leaf]->getScore(variables);
            if (g < succ->getG()) 
	    {
                // the update the information
                succ->setLeaf(leaf);
                succ->setG(g);

                // and the priority queue
                if (succ->getPqPos() == -2)
                {
                	succ->setPqPos(0);
                	openList.push(succ);
                }
                else
                  openList.update(succ);
            }
	    //printf("Line %d, succ update after popping out node u, variables %s, leaf %d, f=%f, openList size %d\n", __LINE__, varsetToString(newVariables).c_str(), leaf, succ->getLeaf(), succ->getF(), openList.size());
        }
    }

    //printf("Line %d, Nodes expanded: %d, open list size: %d\n", __LINE__, nodesExpanded, openList.size());

    heuristic->printStatistics();

    if (goal != NULL) {
        //printf("Found solution: %f\n", goal->getF());
        printf("Line %d, Found solution: %f, scc # %d, netFile.length %lu\n", __LINE__, goal->getG(), scc_idx, netFile.length());

        if (netFile.length() > 0) {
	    network_file.open(netFile.c_str(), std::ios_base::trunc);
            printf("Opened network file %s\n", netFile.c_str());
            datastructures::BayesianNetwork *network = cache.getNetwork();
            network->fixCardinalities();
            std::vector<int> total_ordering(variableCount);
            std::vector<varset> optimalParents;
            std::vector<varset> optimalChildren;
	    std::vector<float> optimalScores;
	    reconstructSolution(goal, spgs, generatedNodes, the_scc, total_ordering, optimalParents, optimalChildren, optimalScores);
            printf("Finished reconstructSolution\n");
            network->setParents(optimalParents);
            printf("Setting uniform probability\n");
            //network->setUniformProbabilities();
            // post processing
//            printf("Start post-processing\n");
//            scoring::ScoringFunction *scoringFunction;
//            if (sf.compare("lasso") == 0 or sf.compare("entropy") == 0){
//                printf("Creating Entropy/LassoFunction with input file %s\n", inputFile.c_str());
//                scoringFunction = new scoring::LassoEntropyScoringFunction(*network, which, inputFile, lambda, NULL, false);
//            } else if (sf.compare("adaptive_lasso") == 0 or sf.compare("adaptive") == 0){
//                printf("Creating Adaptive Entropy/LassoFunction with input file %s\n", inputFile.c_str());
//                scoringFunction = new scoring::AdaptiveLassoEntropyScoringFunction(*network, which, inputFile, lambda, NULL, adaptive, false, &skeleton);
//            }
            // run post_processing
//            scoringFunction->post_processing(total_ordering, optimalParents);
//	        post_astar_processing(spgs,variableCount, total_ordering, optimalParents); // just print network file
	        network_file.close();

            printf("Finished setting parents for network, scc_idx %d %s, count %d, total ordering: ", scc_idx, marker, variableCount);
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
	    const int num_vars = cardinality(allVariables);
	    printf("Recording parents and children into matrix, num_vars = %d\n", num_vars);
            for(int v=0; v < num_vars; v++)
            {
                const varset & parents = optimalParents[v];
                vpar[total_ordering[v]] = parents;
                optimal_parents[total_ordering[v] ] = parents;
                optimal_children[total_ordering[v] ] = optimalChildren[v];
                printf("Variable # %d, parent bit set %lx, children bitset %lx, parents are ", total_ordering[v], parents, optimalChildren[v]);
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
    delete heuristic;

}// run_astar_on_one_scc

// Priority:
// 1. Variables with all neighbors in current cluster
// 2. Number of variables in current cluster
// 3. Number of internal variables in candidate cluster
// 4. Number of new variables in candidate cluster
// Keep all priority canculation in a std::map for future selections
inline uint32_t candidate_priority(const int current_var_idx, const int candidate_var_idx, const varset clusters[], const datastructures::Skeleton & skeleton)
{
  const int variableCount = skeleton.size();
  VARSET_NEW(empty_set, variableCount);

  const varset & orig_cluster = current_var_idx == - 1 ? empty_set : clusters[current_var_idx];
  const varset & candidate_cluster = clusters[candidate_var_idx];
  const varset & neighbors = skeleton.get_neighbors(candidate_var_idx);

  uint32_t all_neighbors_in_current_cluster = VARSET_IS_SUBSET_OF(neighbors, orig_cluster);
  uint32_t num_neighbors_in_current_cluster = cardinality( VARSET_AND(neighbors, orig_cluster) );
  const varset & intersection = VARSET_AND(orig_cluster, candidate_cluster );
  uint32_t num_new_variables = cardinality(candidate_cluster) - cardinality(intersection);
  uint32_t num_internal_variables = 0;
  for(int i=0; i < variableCount; i++)
  {
    num_internal_variables += VARSET_GET(candidate_cluster, i) and VARSET_IS_SUBSET_OF(skeleton.get_neighbors(i), candidate_cluster) ;
  }
  return (all_neighbors_in_current_cluster << 15) + ( num_neighbors_in_current_cluster  << 10 ) + (num_internal_variables << 5) + num_new_variables;
}

// check whether it forms a directed cycle involving u and v, assuming we plan to add directed edge u -> v 
bool has_directed_path(const std::vector<std::vector<int> > & directed_graph, int u, int dest, const std::vector<Neighbors<60> > & vnbs, std::vector<bool> & visited, bool & pre_existing_cycle)
{
  //  will try to find a directed path from v to u
  if(u == dest)
    return true;

  // already visted u, but u is not destination
  if(visited[u]) // some cycle already existed
  {
    pre_existing_cycle = true;
    return false;
  }

  // first time visits u
  visited[u] = true;
  // follow v-structure children 
  for(int i=0; i< vnbs[u].num_vc; i++)
  {
    if(has_directed_path(directed_graph, vnbs[u].vc[i], dest, vnbs, visited, pre_existing_cycle))
      return true;
  }
  /*
  // follow oriented children
  for(int i=0; i< vnbs[u].num_oc; i++)
  {
    if(has_directed_path(directed_graph, vnbs[u].oc[i], dest, vnbs, visited, pre_existing_cycle))
      return true;
  }
  */

  return false;
}

void get_unfaithful_neighbors( const int i
		             , const int j
			     , const varset & opc_i // optimal parents for i
			     , const varset & opc_j // optimal parents for j
			     , const datastructures::Skeleton & skeleton
			     , const int variableCount
			     , std::vector<int> & unfaithful
			     , const int follow_unfaithful
		             )
{
	const varset & nb_i = skeleton.get_neighbors(i);
	const varset & nb_j = skeleton.get_neighbors(j);
	varset opc_i_uf = VARSET_AND(opc_i, VARSET_NOT(nb_i) ); // unfaithful edge with i, 
	varset opc_j_uf = VARSET_AND(opc_j, VARSET_NOT(nb_j) ); // unfaithful edge with j
	unfaithful.clear();
	unfaithful.reserve(variableCount);
	char buf_i[1024]={'\0'}, buf_j[1024]={'\0'};
	int ni = 0;
	int nj = 0;
	for(int k=0; (opc_i_uf or opc_j_uf) and k < variableCount; k++)
	{
          bool edge_i_k = k!=j and VARSET_GET(opc_i_uf, k) > 0;
	  bool edge_j_k = k!=i and VARSET_GET(opc_j_uf, k) > 0;
	  if(edge_i_k)
            ni += snprintf(buf_i + ni, 8, "%d,", k); 
	  if(edge_j_k)
            nj += snprintf(buf_j + nj, 8, "%d,", k); 
	  if(edge_i_k or edge_j_k)
	  {
	    unfaithful.push_back(k);
	    // clear the bit so that we might break loop early
	    VARSET_CLEAR(opc_i_uf, k);
	    VARSET_CLEAR(opc_j_uf, k);
	  }
	}
	//if(buf_i[0])
	//  printf("Found unfaithful edges, follow_unfaithful %d, (%d - %d -): %s \n", follow_unfaithful, j, i, buf_i);
	//if(buf_j[0])
	//  printf("Found unfaithful edges, follow_unfaithful %d, (%d - %d -): %s \n", follow_unfaithful, i, j, buf_j);
}// end get_unfaithful_neighbors

int update_skeleton_and_clusters( int i
		                 , int j
				 , int k
				 , datastructures::Skeleton & skeleton
				 , std::vector<varset> & clusters
				 , std::vector<std::vector<int> > & directed_graph 
				 )
{
    int new_edges = 0;
    if( (directed_graph[i][j] or directed_graph[j][i]) and not VARSET_GET(skeleton.get_neighbors(i), j) )
    {
      skeleton.add_edge(i,j);
      clusters[i] = skeleton.get_neighbors(i);
      clusters[j] = skeleton.get_neighbors(j);
      new_edges++;
    }
    if( (directed_graph[i][k] or directed_graph[k][i]) and not VARSET_GET(skeleton.get_neighbors(i), k) )
    {
      skeleton.add_edge(i,k);
      clusters[i] = skeleton.get_neighbors(i);
      clusters[k] = skeleton.get_neighbors(k);
      new_edges++;
    }
    if( (directed_graph[j][k] or directed_graph[k][j]) and not VARSET_GET(skeleton.get_neighbors(j), k) )
    {
      skeleton.add_edge(j,k);
      clusters[j] = skeleton.get_neighbors(j);
      clusters[k] = skeleton.get_neighbors(k);
      new_edges++;
    }
    return new_edges;
}

void process_triple( const int i
		   , varset & opc_i // optimal parent child for i
		   , const int vj
		   , varset & opc_vj // optimal parent child for vj 
		   , const int vk
		   , varset & opc_vk // optimal parent child for vk
		   , const std::vector<varset> & clusters
		   , std::set<uint64_t> & triplets_checked
		   , const int variableCount
                   , std::vector<bestscorecalculators::BestScoreCalculator*> & spgs
                   , scoring::ScoreCache & cache
		   , const varset & scc
                   , varset & ancestors  // ancestors
                   , const datastructures::EdgeConstraints & constraints
		   , std::vector<varset> & vstr_parents
		   , std::vector<std::vector<int> > & directed_graph
		   , int & num_v_structures
		   , std::vector<std::vector<float> > & bic_scores
		   , const datastructures::Skeleton & skeleton
		   )
{
	uint64_t arr[4];
	arr[0]=i;
	arr[1]=vj;
	arr[2]=vk;
	std::qsort(arr, 3, sizeof arr[0], [](const void *a, const void* b){return *(int*)a - *(int*)b; });
	varset big_cluster = VARSET_OR( VARSET_OR(clusters[i], clusters[vj]), clusters[vk] );
	// If cluster too big, skip
	const int big_cluster_size = cardinality(big_cluster);
	if(big_cluster_size > 26)
	{
	  printf("Not running big cluster for triplet(%lu,%lu,%lu), big_cluster_size = %d\n", arr[0],arr[1],arr[2], big_cluster_size); 
	  return;
	}
	// check whether this triplet has been run before
	uint64_t triplet_key = (arr[0] << 40 ) + (arr[1] << 20) + arr[2]; 
	if(triplets_checked.find(triplet_key) != triplets_checked.end())
	  return;
	char marker[32];
	snprintf(marker, 32, "triplet %lu %lu %lu", arr[0], arr[1], arr[2]);
	triplets_checked.insert(triplet_key);
	printf("Deciding whether to throw away vstructure, triplets_checked %d, center var %d, parents %d %d,(%d->%d,%f),(%d->%d,%f),(%d->%d,%f),(%d->%d,%f)\n"
	      , int(triplets_checked.size()), i, vj, vk
	      , i,vj,bic_scores[i][vj], vj,i,bic_scores[vj][i], i,vk,bic_scores[i][vk], vk,i,bic_scores[vk][i]
	      );
	// run astar before throwing away
        VARSET_NEW(empty_set, variableCount);
	std::vector<varset> op(variableCount, empty_set); // optimal_parent for the tri-cluster
	std::vector<varset> oc(variableCount, empty_set); // optimal_children for the tri-cluster
        run_astar_on_one_scc(variableCount, spgs, cache, scc, ancestors, big_cluster, constraints, i, skeleton, op, oc, marker);
	bool parents_vj_vk = VARSET_GET(op[i], vj) and VARSET_GET(op[i], vk); // vj, vk are parents, i is child
	bool parents_i_vj  = VARSET_GET(op[vk], i) and VARSET_GET(op[vk],vj); // i, vj are parents, vk is child
	bool parents_i_vk  = VARSET_GET(op[vj], i) and VARSET_GET(op[vj],vk); // i, vj are parents, vj is child
	// calculate the direct neighbors for unfaithfulness detection
	opc_i = VARSET_OR(op[i], oc[i]);
	opc_vj= VARSET_OR(op[vj],oc[vj]);
	opc_vk= VARSET_OR(op[vk],oc[vk]);
	printf("Run astar on tri-cluster(%d, %d, %d), center var(%d, %d, %d), throwaway %d, triplets_checked %d\n"
	      , i, vj, vk, parents_vj_vk, parents_i_vk, parents_i_vj
	      , parents_vj_vk || parents_i_vk || parents_i_vj, int(triplets_checked.size())		
	      );

	if(parents_vj_vk) // i is child
	{
          num_v_structures++; 
	  if(0==directed_graph[vj][i] and 1==directed_graph[i][vj])
	    printf("Warning, conflicting edge orientation, already have %d -> %d\n", i, vj);
	  if(0==directed_graph[vk][i] and 1==directed_graph[i][vk])
	    printf("Warning, conflicting edge orientation, already have %d -> %d\n", i, vk);
          directed_graph[vj][i] = 1;
          directed_graph[vk][i] = 1;
          directed_graph[i][vj] = 0;
          directed_graph[i][vk] = 0;
	  // Now checking direction vj - vk
	  if( (VARSET_GET(op[vj], vk) || VARSET_GET(op[vk], vj)) && (0==directed_graph[vk][vj] && 0==directed_graph[vj][vk]) )
          {
            //bool vj_2_vk = VARSET_GET(op[vk], vj);
            directed_graph[vk][vj] = 1; //not vj_2_vk;
            directed_graph[vj][vk] = 1; //vj_2_vk;
	  }
	  VARSET_SET(vstr_parents[i], vk);
	  VARSET_SET(vstr_parents[i], vj);
	  // check additional V-structures
	  varset cc_ij = VARSET_AND(oc[i], oc[vj]);
	  varset cc_ik = VARSET_AND(oc[i], oc[vk]);
	  varset cc_jk = VARSET_AND(oc[vj], oc[vk]);
	  bool has_cc = cc_ij or cc_ik or cc_jk;
	  if(has_cc) 
	    printf("has V-structure as common child (%d, %d, %d)\n", i, vj, vk);
	}
	else if(parents_i_vj) // vk is child
	{
          num_v_structures++; 
	  if(0==directed_graph[vj][vk] and 1==directed_graph[vk][vj])
	    printf("Warning, conflicting edge orientation, already have %d -> %d\n", vk, vj);
	  if(0==directed_graph[i][vk] and 1==directed_graph[vk][i])
	    printf("Warning, conflicting edge orientation, already have %d -> %d\n", vk, i);
          directed_graph[vj][vk] = 1;
          directed_graph[i][vk] = 1;
          directed_graph[vk][i] = 0;
          directed_graph[vk][vj] = 0;
	  if( (VARSET_GET(op[vj], i) || VARSET_GET(op[i], vj)) && (0==directed_graph[i][vj] && 0==directed_graph[vj][i]) )
          {
            //bool i_2_vj = VARSET_GET(op[vj], i);
            directed_graph[i][vj] = 1; //i_2_vj;
            directed_graph[vj][i] = 1; //not i_2_vj;
	  }
	  VARSET_SET(vstr_parents[vk], i);
	  VARSET_SET(vstr_parents[vk], vj);
	  /*
	  if(VARSET_GET(op[i], vj)) // vj is parent of i
	  {
            directed_graph[vj][i] = 1;
            directed_graph[i][vj] = 0;
	  } 
	  else if(VARSET_GET(op[vj], i)) // i is parent of vj
	  {
            directed_graph[vj][i] = 0;
            directed_graph[i][vj] = 1;
	  }
	  */
	}
	else if(parents_i_vk) // vj is child
	{
          num_v_structures++; 
	  if(0==directed_graph[vk][vj] and 1==directed_graph[vj][vk])
	    printf("Warning, conflicting edge orientation, already have %d -> %d\n", vj, vk);
	  if(0==directed_graph[i][vj] and 1==directed_graph[vj][i])
	    printf("Warning, conflicting edge orientation, already have %d -> %d\n", vj, i);
          directed_graph[vk][vj] = 1;
          directed_graph[i][vj] = 1;
          directed_graph[vj][i] = 0;
          directed_graph[vj][vk] = 0;
	  if( (VARSET_GET(op[vk], i) || VARSET_GET(op[i], vk)) && (0==directed_graph[i][vk] && 0==directed_graph[vk][i]) )
          {
            //bool i_2_vk = VARSET_GET(op[vk], i);
            directed_graph[i][vk] = 1; //i_2_vk;
            directed_graph[vk][i] = 1; //not i_2_vk;
	  }
	  VARSET_SET(vstr_parents[vj], i);
	  VARSET_SET(vstr_parents[vj], vk);
	  /*
	  if(VARSET_GET(op[i], vk)) // vk is parent of i
	  {
            directed_graph[vk][i] = 1;
            directed_graph[i][vk] = 0;
	  } 
	  else if(VARSET_GET(op[vk], i)) // i is parent of vk
	  {
            directed_graph[vk][i] = 0;
            directed_graph[i][vk] = 1;
	  }
	  */
	}
	else
	{
	  if( (VARSET_GET(op[vj], vk) || VARSET_GET(op[vk], vj)) && (0==directed_graph[vk][vj] && 0==directed_graph[vj][vk]) )
          {
            directed_graph[vk][vj] = 1;
            directed_graph[vj][vk] = 1;
	  }
	  if( (VARSET_GET(op[vj], i) || VARSET_GET(op[i], vj)) && (0==directed_graph[i][vj] && 0==directed_graph[vj][i]) )
          {
            directed_graph[i][vj] = 1;
            directed_graph[vj][i] = 1;
	  }
	  if( (VARSET_GET(op[vk], i) || VARSET_GET(op[i], vk)) && (0==directed_graph[i][vk] && 0==directed_graph[vk][i]) )
          {
            directed_graph[i][vk] = 1;
            directed_graph[vk][i] = 1;
	  }
	  printf("No V-structure for triplet(%lu,%lu,%lu), parents_%d(%d,%d), parents_%d(%d,%d), parents_%d(%d,%d)\n"
	        , arr[0], arr[1], arr[2]
		, i, (VARSET_GET(op[i], vj) ? vj : -1), (VARSET_GET(op[i], vk) ? vk : -1)
		, vj, (VARSET_GET(op[vj], vk) ? vk : -1), (VARSET_GET(op[vj], i) ? i : -1)
		, vk, (VARSET_GET(op[vk], vj) ? vj : -1), (VARSET_GET(op[vk], i) ? i : -1)
		);
	}
}//process_triple

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
    int max_scc_size = 0;
    for(int i=0; i < num_scc; i++)
    {
      const int size_scc = cardinality(scc_list[i]);
      if(size_scc > max_scc_size)
	max_scc_size = size_scc;
    }
    printf("num of sccs = %d, max_scc_size = %d\n", num_scc, max_scc_size);
    act.start();

//    std::vector<int> sequence;
//    sequence.resize(variableCount);
//    for(int i=0; i<variableCount; i++)
    VARSET_NEW(empty_set, variableCount);
    std::vector<varset> clusters(variableCount, empty_set);
    std::vector<varset> larger_clusters(variableCount, empty_set);
    std::vector<int> larger_cardinality(variableCount, 0);
    //const int num_variables_per_cluster = 32;
    //skeleton.bfs_partition(clusters, num_variables_per_cluster, sequence);
    // easy partition, just MB neighbors
    for(int i=0; i<variableCount; i++)
    {
    	varset neighbors = skeleton.get_neighbors(i);
	clusters[i] = neighbors;
	larger_clusters[i] = neighbors;
	VARSET_SET(clusters[i], i);
    	std::cout << "Printing Markov neighbors, Var " << i
    		  << ", neighbors " << varsetToString(neighbors)
		  << ", cluster " << varsetToString(clusters[i])
		  << ", contained " << VARSET_IS_SUBSET_OF(neighbors, clusters[i])
                  << std::endl;
    }
    for(int i=0; i<variableCount; i++)
    {
	for(int j=0; j<variableCount; j++)
	{
	  if(VARSET_GET(clusters[i], j))
	    larger_clusters[i] = VARSET_OR(larger_clusters[i], clusters[j]);
	}
    }
    for(int i=0; i<variableCount; i++)
    {
       larger_cardinality[i] = cardinality(larger_clusters[i]);
       std::cout << "Larger_cluster around variable " << i
	         << ", cluster size " << larger_cardinality[i]
		 << std::endl
	         ;
    }
    fflush(NULL);

    datastructures::EdgeConstraints constraints(variableCount);
    std::vector<std::vector<varset> > optimal_parents(variableCount);
    std::vector<std::vector<varset> > optimal_children(variableCount);
    std::vector<std::vector<int> > directed_graph(variableCount);
    std::vector<std::vector<float> > bic_scores(variableCount); // bic_scores[i][j] is the score for edge i->j, score(j|i)
    std::vector<std::vector<float> > lasso_scores(variableCount); // bic_scores[i][j] is the score for edge i->j, score(j|i)
    for(int i=0; i<variableCount; i++)
    {
    	directed_graph[i] = std::vector<int>(variableCount, 0);
	bic_scores[i]  = std::vector<float>(variableCount, 0);  
	lasso_scores[i]  = std::vector<float>(variableCount, 0);
    }
    // now check each edge
    std::vector<varset> astar_neighbors(variableCount, empty_set);

    int num_v_structures = 0;
    // pre-calculate scores between undirected pairs
    for(int i=0; i<variableCount; i++)
    {
	for(int j=0; j<i; j++)
	{
	   // check Lasso BIC direction
           varset p = empty_set; // parent consisting of i 
           VARSET_SET(p, i);
           // score_j_given_i 
	   bic_scores[i][j] = spgs[j]->getScore(p);
	   p = empty_set;
	   VARSET_SET(p, j);
           // score_i_given_j 
	   bic_scores[j][i] = spgs[i]->getScore(p);
	   //if(bic_scores[j][i] > 1e-6 or bic_scores[i][j] > 1e-6)
	   //  printf("Calculated pair-wise edge score, bic_scores(%d|%d)=%.6f, bic_scores(%d|%d)=%.6f, equal_score %d\n", j, i, bic_scores[i][j], i,j,bic_scores[j][i], equal_score );
	}
    }

    // Now run Astar on uncertain triplets
    bool follow_more_unfaithful = false;
    std::set<uint64_t> triplets_checked; // use 20 bits for each variable index
    std::vector<varset> vstr_parents(variableCount, 0L); // V-structure parents, definite
    //for(int iteration=0; iteration<2; iteration++)
    //{
    for(int i=0; i<variableCount; i++)
    {
	const varset parents_i_in_cluster_i = skeleton.get_neighbors(i);
	std::vector<int> parents, uncertain_parents;
        parents.reserve(variableCount);
	uncertain_parents.reserve(variableCount);
	// Add v-structure combinations
//	for(int j=0; j<variableCount; j++)
//	{
//	  if(VARSET_GET(parents_i_in_cluster_i, j) and directed_graph[i][j] == directed_graph[j][i] ) // only use the undirected i-j edges 
//	  {
//	    uncertain_parents.push_back(j);
//	  }
//	}
        // Add v-structure combinations
        for(int j=0; j<variableCount; j++)
        {
          if(VARSET_GET(parents_i_in_cluster_i, j))
          {
            parents.push_back(j);
            //if(directed_graph[i][j] == directed_graph[j][i])
              uncertain_parents.push_back(j);
            //else if(directed_graph[j][i] == 1) // j has i as child
            //  constraints.add_ancestors(i, j);
          }
        }
        printf("Certain and uncertain parents, variable %d, num_sure_parents %d, num_uncertain_parents %d\n", i, int(parents.size()), int(uncertain_parents.size()));
        // if there is only one uncertain, we should add one that is certain
        if(uncertain_parents.size() == 1 and parents.size() > 1)
        {
          for(size_t m=0; m < parents.size(); m++)
          {
            if(parents[m] != uncertain_parents[0])
            {
              uncertain_parents.push_back(parents[m]);
              printf("Only one uncertain parent, add one for triplet, variable %d, sure parent %d, uncertain parent %d\n", i, parents[m], uncertain_parents[0]);
              break;
            }
          }
        }
	else if(uncertain_parents.size() == 1 and parents.size() == 1 and i < parents[0] and 1==cardinality(skeleton.get_neighbors(parents[0])) )
	{
	  int vj = uncertain_parents[0];
	  VARSET_NEW(parset, variableCount);
	  VARSET_SET(parset, vj);
	  float score = spgs[i]->getScore(parset);
          varset the_parents = spgs[i]->getParents();
	  if(VARSET_EQUAL(the_parents, parset))
	  {
	    printf("Found isolated orphan edge %d - %d\n", i, vj);
	    directed_graph[i][vj]=1;
	    directed_graph[vj][i]=1;
	  }
	}

	printf("uncertain parents, variable %d, num_uncertain_parents %d\n", i, int(uncertain_parents.size()));
	for(size_t j=0; j < uncertain_parents.size(); j++)
	{
	  int vj = uncertain_parents[j];
	  for(size_t k=0;k<j; k++)
	  {
	    int vk = uncertain_parents[k];
	    varset opc_i, opc_vj, opc_vk; // Optimal parent child
	    process_triple( i, opc_i, vj, opc_vj, vk, opc_vk, clusters, triplets_checked, variableCount, spgs, cache, scc, ancestors, constraints, vstr_parents
			  , directed_graph, num_v_structures, bic_scores, skeleton );
	    // check whether edge vj-vk exists and is unfaithful
	    if(not VARSET_GET(skeleton.get_neighbors(vj), vk) and (directed_graph[vj][vk] or directed_graph[vk][vj]) )
	    {
		skeleton.add_edge(vj,vk);
                clusters[vj] = skeleton.get_neighbors(vj);
                clusters[vk] = skeleton.get_neighbors(vk);
		printf("Found unfaithful edge vjk, root var %d, added (%d,%d) into skeleton\n", i, vj, vk);
	    }
	    if(not follow_more_unfaithful) continue;
	    // check unfaithful edges
	    std::vector<int>  unfaithful_i_vj;
            std::vector<int>  unfaithful_i_vk;
            std::vector<int>  unfaithful_vj_vk;
            int follow_unfaithful = 0;
	    get_unfaithful_neighbors( i, vj, opc_i, opc_vj, skeleton, variableCount, unfaithful_i_vj,  follow_unfaithful);
	    get_unfaithful_neighbors( i, vk, opc_i, opc_vk, skeleton, variableCount, unfaithful_i_vk,  follow_unfaithful);
	    get_unfaithful_neighbors( vj,vk, opc_vj,opc_vk, skeleton, variableCount, unfaithful_vj_vk, follow_unfaithful);
	    int unfaithful_count = unfaithful_i_vj.size() + unfaithful_i_vk.size() + unfaithful_vj_vk.size();
	    if( 0 == unfaithful_count) 
	      continue;

	    int unfaithful_edges = update_skeleton_and_clusters( i, vj, vk, skeleton, clusters, directed_graph);

	    for(size_t u=0; u < unfaithful_i_vj.size(); u++)
	    {
	        const int vu = unfaithful_i_vj[u];
		VARSET_NEW(opc_vu, variableCount);
	        process_triple( i, opc_i, vj, opc_vj, vu, opc_vu, clusters, triplets_checked, variableCount, spgs, cache, scc, 
			    ancestors, constraints, vstr_parents, directed_graph, num_v_structures, bic_scores, skeleton);
                unfaithful_edges += update_skeleton_and_clusters( i, vj, vu, skeleton, clusters, directed_graph);
	    }
	    for(size_t u=0; u < unfaithful_i_vk.size(); u++)
	    {
		const int vu = unfaithful_i_vk[u];
		VARSET_NEW(opc_vu, variableCount);
	        process_triple( i, opc_i, vk, opc_vj, vu, opc_vu, clusters, triplets_checked, variableCount, spgs, cache, scc, 
			    ancestors, constraints, vstr_parents, directed_graph, num_v_structures, bic_scores, skeleton);
                unfaithful_edges += update_skeleton_and_clusters( i, vk, vu, skeleton, clusters, directed_graph);
	    }
	    for(size_t u=0; u < unfaithful_vj_vk.size(); u++)
	    {
		const int vu = unfaithful_vj_vk[u];
		VARSET_NEW(opc_vu, variableCount);
	        process_triple( vj, opc_vj, vk, opc_vk, vu, opc_vu, clusters, triplets_checked, variableCount, spgs, cache, scc, 
			    ancestors, constraints, vstr_parents, directed_graph, num_v_structures, bic_scores, skeleton);
                unfaithful_edges += update_skeleton_and_clusters( vj, vk, vu, skeleton, clusters, directed_graph);
	    }
	    if(unfaithful_edges)
	      printf("Checking unfaithful, orig triple(%d,%d,%d), unfaithful_count %d, added unfaithful edges %d, follow_unfaithful = %d, (%d,%d):%lu, (%d,%d):%lu, (%d,%d):%lu\n"
	      , i, vj, vk, unfaithful_count, unfaithful_edges, follow_unfaithful, i, vj, unfaithful_i_vj.size(), i, vk, unfaithful_i_vk.size(), vj, vk, unfaithful_vj_vk.size() 
	      );
	  }// for k
	}// for j
    }//for i
    //}// for iteration

    int unfaithful_count = 0;
    int unfaithful_count_delta = 0;
    int unfaithful_iters = 0;
    do
    {
    unfaithful_count_delta = 0;
    // check whether there is any newly found edge missing in skeleton
    int unfaithful_vstr = 0;
    for(int i=0; i < variableCount; i++)
    {
	const varset & neighbors = skeleton.get_neighbors(i);
	for(int j=0; j < i; j++)
	{
	    bool unfaithful = (directed_graph[i][j] or directed_graph[j][i] ) and 0==VARSET_GET(neighbors, j);
	    unfaithful_count += unfaithful;
	    unfaithful_count_delta += unfaithful;
	    if(not unfaithful)
	      continue;
	    skeleton.add_edge(i,j); 
            clusters[i] = skeleton.get_neighbors(i);
            clusters[j] = skeleton.get_neighbors(j);
	    const int old_num_vstr = num_v_structures;
	    // check around i
	    varset opc_i, opc_j, opc_k; // Optimal parent child
	    for(int k=0; k < variableCount; k++)
	    {
	      if(k != i and k != j and (VARSET_GET(clusters[i], k) or VARSET_GET(clusters[j], k) ) )
	        process_triple( i, opc_i, j, opc_j, k, opc_k, clusters, triplets_checked, variableCount, spgs, cache, scc, ancestors, constraints, vstr_parents, 
				directed_graph, num_v_structures, bic_scores, skeleton
			      );
	    }
	    unfaithful_vstr += num_v_structures - old_num_vstr;
	}
    }
    if(unfaithful_count_delta > 0)
      printf("Found unfaithful edges, unfaithful edge count %d, unfaithful_vstr %d, iteration %d\n", unfaithful_count, unfaithful_vstr, unfaithful_iters );
    unfaithful_iters++;
    }while(unfaithful_count_delta > 0);

    // Apply the orientation rules
    // first gather all unorientied edges
    // apply rule 2
  int total_oriented = 0;
  int iter = 0;
  for(; iter < variableCount; iter++)
  {
    int num_oriented = 0;
    // apply Rule 2
    for(int v = 0; v < variableCount; v++)
    {
        // check whether some edges in and some edges out
        std::vector<int> ins, outs;
        ins.reserve(variableCount);
        outs.reserve(variableCount);
        for(int j=0; j < variableCount; j++)
        {
          if(1==directed_graph[v][j] and 0==directed_graph[j][v])
            outs.push_back(j);
          else if(1==directed_graph[j][v] and 0==directed_graph[v][j])
            ins.push_back(j);
        }
	if(0==ins.size() or 0==outs.size()) continue;
	printf("Checking Meek rule 2, iteration %d, root var %d, num_ins %lu, num_outs %lu\n"
	      , iter, v, ins.size(), outs.size()); 
	for(size_t j=0; j<ins.size(); j++)
	{
	  int parent = ins[j];
	  for(size_t k=0; k < outs.size();  k++)
	  {
	    int child = outs[k];
	    if(directed_graph[parent][child] and directed_graph[child][parent])
	    {
		directed_graph[parent][child] = 1;
		directed_graph[child][parent] = 0;
		num_oriented++;
		printf("Applied Meek rule 2, iteration %d, oriented parent=%d -> child=%d, mid var=%d, num_oriented %d\n"
		      , iter, parent, child, v,  num_oriented);
	    }
	  }
	}
    }// for each v, rule 2

    // apply Meek rule #3, orient diamond shape
    for(int v=0; v<variableCount; v++)
    {
	int num_vps = cardinality(vstr_parents[v]); // num of v-structure parents
	std::string vps = varsetToString(vstr_parents[v]);
	std::vector<int> vp, undirected;
	vp.reserve(variableCount);
	undirected.reserve(variableCount);
	for(int j=0; j<variableCount; j++)
	{
	  if(VARSET_GET(vstr_parents[v], j))
	    vp.push_back(j);
	  if(1 == directed_graph[v][j] and 1==directed_graph[j][v])
	    undirected.push_back(j);
	}

	if(num_vps < 2 or 0==undirected.size() ) continue;
	printf("Checking Meek rule 3, iteration %d, root var %d, num_ins %lu, num_undirected %lu\n"
	      , iter, v, vp.size(), undirected.size()); 

	for(size_t j=0; j<undirected.size(); j++)
	{
	  int neighbor = undirected[j];
	  // Now 1==directed_graph[v][neighbor] and 1==directed_graph[neighbor][v]
	  // check whether neighbor, v, and v's parents form a diamond
	  int num_par_connected = 0;
	  char buf[1024];
	  int N=0;
	  for(size_t k=0; k < vp.size(); k++)
	  {
	    int par = vp[k];
	    if(1==directed_graph[par][neighbor] and 1==directed_graph[neighbor][par] )
	    {
	      num_par_connected++;
	      N += snprintf(buf + N, 16, ",%d", par);
	    }
	  }
	  if(num_par_connected >= 2) // form a diamond, set j -> v
	  {
	    directed_graph[v][neighbor] = 0;
	    num_oriented++;
	    printf("Applied Meek rule 3 to root variable %d, new parent %d, v-structure parents: %s, diamond parents: %s, num_oriented %d, iteration %d\n"
		  , v, neighbor, vps.c_str(), buf, num_oriented, iter);
	  }
	}
    } // for each v, rule 3

    // apply Meek rule #4, orient diamond shape
    for(int v=0; v < variableCount; v++)
    {
	// No original v-structure from A*
	// check whether some edges in and some edges out
	std::vector<int> ins, outs, undirected;
	ins.reserve(variableCount);
	outs.reserve(variableCount);
	undirected.reserve(variableCount);
	for(int j=0; j < variableCount; j++)
	{
	  if(1==directed_graph[v][j] and 0==directed_graph[j][v])
	    outs.push_back(j);
	  else if(1==directed_graph[j][v] and 0==directed_graph[v][j])
	    ins.push_back(j);
	  else if(1==directed_graph[v][j] and 1==directed_graph[j][v])
	    undirected.push_back(j);
	}
	if(0==outs.size() or 0==ins.size() or 0==undirected.size()) continue; 
	printf("Checking Meek rule 4, iteration %d, root var %d, num_ins %lu, num_outs %lu, num_undirected %lu\n"
	      , iter, v, ins.size(), outs.size(), undirected.size()); 

	// Now some in, some out, some undirected
	// for each undirected edge
	for(size_t k=0; k< undirected.size(); k++)
	{
	   int neighbor = undirected[k];
	   // check whether neighbor is linked to the in- and out- edges to form diamond
	   std::vector<int> diamond_ins, diamond_outs;
	   diamond_ins.reserve(ins.size());
	   diamond_outs.reserve(outs.size());
	   char buf_ins[256];
	   int N_in = 0;
	   for(size_t j=0; j<ins.size(); j++)
	   {
	     int parent = ins[j];
	     if(1==directed_graph[neighbor][parent] and 1==directed_graph[parent][neighbor])
	     {
	       diamond_ins.push_back(parent);
               N_in += snprintf(buf_ins + N_in, 16, ",%d", parent);
	     }
	   }
	   // if no undirected edge with any parent of v, skip
	   if( 0 == diamond_ins.size() ) continue;
	   for(size_t j=0; j<outs.size(); j++)
	   {
	     int child = outs[j];
	     if(0==directed_graph[neighbor][child] or 0==directed_graph[child][neighbor])
	       continue;
	     num_oriented++;
	     // Now neighbor <-> child
	     directed_graph[child][neighbor] = 0;
	     printf("Applied Meek rule 4, added edge %d -> %d, bottom variable %d, neighbor %d undirected with %s, num_oriented %d, iteration %d\n",
	            neighbor, child, v, neighbor, buf_ins, num_oriented, iter);
	   }
	}// for k

    }// for each v, rule 4

    /*
    // Rule 1 start here
    // Apply rule 1
    for(int i=0; i<variableCount; i++)
    {
      std::vector<int> ins;
      std::vector<int> outs;
      std::vector<int> undirected;
      for(int j=0; j<variableCount; j++)
      {
	if(1==directed_graph[i][j] and 0==directed_graph[j][i]) // i->j
	  ins.push_back(j);
	else if(1==directed_graph[j][i] and 0==directed_graph[i][j]) // i <- j
	  outs.push_back(j);
	else if(1==directed_graph[j][i] and 1==directed_graph[i][j]) // i <-> j
	  undirected.push_back(j);
      }
      // one in , orient the undirected outward
      if(1==ins.size() and 1==undirected.size())
      {
        for(size_t k=0; k<undirected.size(); k++)
	{
          int vk = undirected[k];
	  directed_graph[vk][i] = 0; // already have directed_graph[i][vk] ==1
	  num_oriented++;
	  printf("Applied Meek rule 1, oriented edge %d -> %d, num_oriented %d, iteration %d\n", i, vk, num_oriented, iter); 
	}	
      }

    }// for each variable v
    // Rule 1 ends here
    */

    total_oriented += num_oriented;
    if(0==num_oriented)
      break;
  }// iterating
  printf("Total oriented edges %d, iterations %d\n", total_oriented, iter);

 /*
    while(ud.has_more_edge()) // for each undirected edge
    {
      int v1, v2;
      ud.next_edge(v1, v2);
      varset v1_nbs = skeleton.get_neighbors(v1);
      varset v2_nbs = skeleton.get_neighbors(v2);
      bool both_in_mb = VARSET_GET(v1_nbs, v2) and VARSET_GET(v2_nbs, v1);
      if(not both_in_mb)
      {
	printf("Not both_in_mb, removed undirected edge(%d, %d)\n", v1, v2);
	directed_graph[v1][v2] = 0;
	directed_graph[v2][v1] = 0;
	continue;
      }
      varset p = empty_set; // parent consisting of i 
      VARSET_SET(p, v1);
      FloatMap cache_tmp;
      // check edge vj - i, and vk -i using the lasso BIC score
      float score_v2_given_v1 = scoringFunction->calculateScore(v2, p, cache_tmp);
      VARSET_CLEAR(p, v1); // set bit i zero
      VARSET_SET(p, v2);
      float score_v1_given_v2 = scoringFunction->calculateScore(v1, p, cache_tmp);
      const double threshold = 10.0; // 6.0 for positive, 10.0 for strong
      const bool score_v1_to_v2_better = score_v2_given_v1 > score_v1_given_v2 ;
      printf("Trying to orient edge (%d, %d), num_undirected_edges %d, score_%d_to_%d_better %d, score(%d|%d)=%f, score(%d|%d)=%f\n", v1, v2, ud.size(), v1, v2, score_v1_to_v2_better
		     , v2,v1,score_v2_given_v1, v1, v2, score_v1_given_v2 );
      // Rule 1
      int pv1=-1;
      int which=-1;
      bool has_i_to_v1 = vnbs[v1].num_vp > 0;
      bool has_i_to_v2 = vnbs[v2].num_vp > 0;
      if(has_i_to_v1 == has_i_to_v2) // Both have directed edge into them, or neither one has
      {
	printf("Not applying rule 1 because both have directed edges, undirected edge(%d,%d), first parents(%d,%d), #parents(%d,%d)\n", v1,v2, vnbs[v1].vp[0], vnbs[v2].vp[0], vnbs[v1].num_vp,vnbs[v2].num_vp);
      }
      else // apply Rule 1
      {
        if(has_i_to_v1) // only v1 has parent, orient v1 -> v2
        {
	printf("Apply rule 1 and oriented %d -> %d\n", v1, v2);
        directed_graph[v1][v2] = 1;	
        directed_graph[v2][v1] = 0;	
	vnbs[v1].add_oc(v2);
	vnbs[v2].add_op(v1);
        }
        else if(has_i_to_v2) // only v2 has parent, orient v2 -> v1
        {
	printf("Apply rule 1 and oriented %d -> %d\n", v2, v1);
        directed_graph[v1][v2] = 0;	
        directed_graph[v2][v1] = 1;	
	vnbs[v2].add_oc(v1);
	vnbs[v1].add_op(v2);
        }
	continue;
      }

      // apply Rule 2
      // Need find a third point v where v1->v->v2 or v2->v->v1
      bool v1_v_v2 = false;
      bool v2_w_v1 = false;
      int mid_v1_v2 = -1;
      int mid_v2_v1 = -1;
      for(int i=0;i<vnbs[v1].num_vc; i++)
      {
	int v = vnbs[v1].vc[i];
	if(1 == directed_graph[v][v2] and 0 == directed_graph[v2][v]) // v1->v->v2
	{
	  v1_v_v2 = true;
	  mid_v1_v2 = v;
	}
      }
      for(int i=0; i< vnbs[v2].num_vc; i++)
      {
	int w = vnbs[v2].vc[i];
	if(1 == directed_graph[w][v1] and 0 == directed_graph[v1][w]) // v2 -> w -> v1
        {
	  v2_w_v1 = true;
	  mid_v2_v1 = w;
	}
      }
      if(v1_v_v2 and not v2_w_v1 and score_v1_to_v2_better) // v1 -> v -> v2, orient v1 -> v2
      {
        directed_graph[v1][v2] = 1;
        directed_graph[v2][v1] = 0;
	vnbs[v1].add_oc(v2);
	vnbs[v2].add_op(v1);
	printf("Apply rule 2a and oriented %d -> %d, mid_v %d, score_%d_to_%d_better %d, score(%d|%d)=%f, score(%d|%d)=%f\n", v1, v2, mid_v1_v2, v1, v2, score_v1_to_v2_better
			, v2,v1,score_v2_given_v1, v1, v2, score_v1_given_v2);
      }
      else if(v2_w_v1 and not v1_v_v2 and not score_v1_to_v2_better)
      {
        directed_graph[v1][v2] = 0;
        directed_graph[v2][v1] = 1;
	vnbs[v2].add_oc(v1);
	vnbs[v1].add_op(v2);
	printf("Apply rule 2b and oriented %d -> %d, mid_v %d, score_%d_to_%d_better %d, score(%d|%d)=%f, score(%d|%d)=%f\n", v2, v1, mid_v2_v1, v1, v2, score_v1_to_v2_better
			, v2,v1,score_v2_given_v1, v1, v2, score_v1_given_v2);
      }
    }// while loop, for each undirected edge
    int num_oc = 0;
    int num_op = 0;
    for(int i=0; i<variableCount; i++)
    {
	num_oc += vnbs[i].num_oc;
	num_op += vnbs[i].num_op;
	for(int j=0; j<vnbs[i].num_op; j++)
	  printf("Oriented edges into %d from parent %d\n", i, vnbs[i].op[j]);
	for(int j=0; j<vnbs[i].num_oc; j++)
	  printf("Oriented edges from %d to child %d\n", i, vnbs[i].oc[j]);
    }
    // apply rule 2 first

    printf("Before printing equivalent graph, num_conflict_v_structure_edges = %d, num_v_structures=%d, num_oriented_parents %d, num_oriented_children %d\n"
          , num_conflict_v_structure_edges
	  , num_v_structures
	  , num_op
	  , num_oc
	  );
*/
    // print the directed_graph
    network_file_csv.open((netFile+".csv").c_str(), std::ios_base::trunc);
    for(int i=0; i<variableCount; i++)
    {
    	std::cout << directed_graph[i][0];
    	network_file_csv << directed_graph[i][0];
    	for(int j=1; j<variableCount; j++)
    	{
    		std::cout << "," << directed_graph[i][j] ;
    		network_file_csv << "," << directed_graph[i][j] ;
    	}
    	network_file_csv << std::endl;
	std::cout << std::endl;
    }

    for (auto spg = spgs.begin(); spg != spgs.end(); spg++) {
        delete *spg;
    }
    //delete heuristic;
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




