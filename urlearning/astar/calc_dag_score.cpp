#include <boost/tokenizer.hpp>
#include <iostream>
#include "urlearning/score_cache/score_cache.h"
#include "urlearning/score_cache/best_score_calculator.h"
#include "urlearning/score_cache/best_score_creator.h"
#include "urlearning/base/typedefs.h"

using namespace std;

std::pair<float,float>
get_dag_score( const char* model_filename
                   , std::vector<bestscorecalculators::BestScoreCalculator*> & spgs 
		   , bool verbose
		   , int & num_edges
		   , int & num_edges_to_remove
		   , int & num_edges_to_remove_alt
                   )
{
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> sep(", \n\r");
    std::ifstream input_file(model_filename, std::ios_base::in);
    if(not input_file.good())
    {
        cerr << "Invalid model file " << model_filename << endl;
        return std::pair<float,float>(0.0,0.0);
    }
    std::string line;
    int varIdx = 0;
    std::getline(input_file, line);
    tokenizer toks(line,sep);
    int variableCount = 0;
    for(auto iter=toks.begin(); iter != toks.end(); iter++)
        variableCount++;
    int edges[64][64];
    for(int i=0; i<variableCount; i++)
        for(int j=0; j<variableCount; j++)
	    edges[i][j] = 0;
    float total_score = 0.0;
    num_edges = 0;
    num_edges_to_remove = 0;
    num_edges_to_remove_alt = 0;
    std::vector<uint64_t> alt_parents(variableCount, 0);
    char msg[1024] = {0};
    char msg_alt[1024] = {0};
    int N = 0;
    int N_alt = 0;
    do{
        uint64_t parents = 0;
        tokenizer tok(line,sep);
        int num=0;
        for(auto iter=tok.begin(); iter != tok.end(); iter++)
        {
                float value = atof(iter->c_str());
                if(std::abs(value) > 1e-5)
		{
                  parents |=  uint64_t(1) << num  ; // variable #num is a parent of variable #varIdx
		  num_edges++;
		  alt_parents[num] |= uint64_t(1) << varIdx;
		  edges[varIdx][num] = 1;
		  edges[num][varIdx] = 1;
		}
                num++;
        }
	// if score available
	if(spgs[varIdx])
	{
	  total_score += spgs[varIdx]->getScore(parents) ;
	  uint64_t optimal_parents = spgs[varIdx]->getParents();
	  uint64_t bit_diff = parents ^ optimal_parents;
	  int diff_bits = cardinality(bit_diff);
	  num_edges_to_remove +=  diff_bits;
	  if(diff_bits)
	  {
	    for(int j=0; j<variableCount;j++)
	    {
		uint64_t mask = uint64_t(1) << j;
		if(bit_diff & mask) 
		  N += snprintf(msg+N, 24, "(%d -> %d),", j, varIdx);
	    }
	  }
	}
        varIdx++;
    }while(std::getline(input_file, line) and varIdx < variableCount );
    input_file.close();

    float alt_score = 0.0;
    for(int i=0; i<variableCount; i++)
    {
	if(spgs[i])
	{
	  alt_score += spgs[i]->getScore(alt_parents[i]); 
	  uint64_t optimal_parents = spgs[i]->getParents();
	  uint64_t bit_diff = alt_parents[i] ^ optimal_parents;
	  int diff_bits = cardinality(bit_diff);
	  num_edges_to_remove_alt += diff_bits;
          if(diff_bits)
          {
            for(int j=0; j<variableCount;j++)
            {
                uint64_t mask = uint64_t(1) << j;
                if(bit_diff & mask)
                  N_alt += snprintf(msg_alt+N_alt, 24, "(%d -> %d),", j, i);
            }
          }
	}
    }
    num_edges = 0;
    for(int i=0; i<variableCount; i++)
	for(int j=i+1; j<variableCount; j++)
	    num_edges += edges[i][j]; 


    if(verbose and total_score < alt_score and msg[0])
      printf("\nSuggest_reg to remove edges %d, scores reg %f alt %f, %s in %s\n", num_edges_to_remove, total_score, alt_score, msg, model_filename);
    else if(verbose and alt_score < total_score and msg_alt[0])
      printf("\nSuggest_alt to remove edges %d, scores reg %f alt %f, %s in %s\n", num_edges_to_remove_alt,total_score, alt_score,  msg_alt, model_filename);
    
    return std::pair<float,float>(total_score, alt_score);
}


// read score file .pss, and multiple DAGs, print score for each DAG
//
int main(int argc, char** argv)
{
  const std::string score_file_name ( argv[1] );

  char** dag_files = argv + 2;
  const int num_models = argc - 2;

  std::ifstream sf(score_file_name);
  const bool score_file_good = sf.good();
  if(score_file_good)
    sf.close();
  scoring::ScoreCache cache;
  if(score_file_good)
    cache.read(score_file_name);

  std::string bestScoreCalculator = "list";
  std::vector<bestscorecalculators::BestScoreCalculator*> spgs(60,NULL);
  if(score_file_good)
    spgs = bestscorecalculators::create(bestScoreCalculator, cache);

  std::vector<std::string> models( dag_files, dag_files + num_models );
  for(int i=0; i< num_models; i++)
  {
        if(models[i].rfind('/') != std::string::npos)	
          models[i] = models[i].substr(models[i].rfind('/')+1);
	if(models[i].find(".csv") != std::string::npos)
	  models[i] = models[i].substr(0,models[i].find(".csv"));
	
        transform(models[i].begin(), models[i].end(), models[i].begin(), ::toupper);
  }


  bool verbose = false;
  float scores[num_models] = {0.0};
  for(int i=0; i<num_models; i++)
  {
    int num_edges = 0; 
    int num_edges_to_remove = 0;
    int num_edges_to_remove_alt = 0;
    std::pair<float,float> dag_scores = get_dag_score(dag_files[i], spgs, verbose, num_edges, num_edges_to_remove, num_edges_to_remove_alt);

    scores[i] = dag_scores.first > dag_scores.second ? dag_scores.second : dag_scores.first ;
    int to_remove = dag_scores.first > dag_scores.second ? num_edges_to_remove : num_edges_to_remove_alt ; 

    //float diff = scores[i] - scores[0];
    //float rel_diff = scores[0] ? diff/abs(scores[0]) : 0.0;
    if( 0 == i )
      printf("%s  %f  edges %d  ", models[i].c_str(), scores[i], num_edges);
    else
      printf("%s  %f  edges %d remove %d ", models[i].c_str(), scores[i], num_edges, to_remove);
  } 
  printf("\n");
}
