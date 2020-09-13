/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "skeleton.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <queue>
#include <boost/tokenizer.hpp>

namespace datastructures {
    
Skeleton::Skeleton(std::string input_filename):initialized(false),variableCount(1),all_bit_set(varset(variableCount))
{
    read_matrix_file(input_filename);
}
bool Skeleton::read_arc_list_file(std::string input_filename, int num_vertices)
{
    std::ifstream input_file(input_filename.c_str(), std::ios_base::in);
    if(not input_file.good())
        return false;
    
    std::cout << __FILE__ << ":" << __LINE__<< ", Skeleton filename: " << input_filename << std::endl;
    VARSET_NEW(empty, num_vertices);
    edges = std::vector<varset>(num_vertices, empty);
    const_cast<int&>(variableCount) = num_vertices;
    // set all bit to 1
    const_cast<varset&>(all_bit_set) = varset(num_vertices);
    VARSET_SET_ALL(const_cast<varset&>(all_bit_set), num_vertices);
    
    std::string line;
    
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> sep(",");
    while(std::getline(input_file, line))
    {
    	tokenizer tok(line,sep);
    	auto iter = tok.begin();
    	std::string from = *iter++;
    	std::string to = *iter;
    	int v1 = atoi(from.c_str()+2);
    	int v2 = atoi(to.c_str()+2);
    	std::cout << "read_arc_list, from = " << from
    			  << ", to = " << to
				  << ", v1 = " << v1
				  << ", v2 = " << v2
				  << std::endl;
    	add_edge(v1-1,v2-1);
    }
    
    input_file.close();
    const_cast<bool&>(initialized) = true;
    calculate_scc();
    return true;
}
bool Skeleton::read_matrix_file(std::string input_filename)
{
    int num_vertices;
    std::ifstream input_file(input_filename.c_str(), std::ios_base::in);
    if(not input_file.good())
        return false;
    std::cout << __FILE__ << ":" << __LINE__<< ", Skeleton filename: " << input_filename << std::endl;
    std::string line;
    std::getline(input_file, line);
    std::stringstream ss(line);
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> sep(", \n\r");
    tokenizer tok(line,sep);
    num_vertices = 0;
    for(auto iter=tok.begin(); iter != tok.end(); iter++)
        num_vertices++;
    std::cout << "tok.size = " << num_vertices << std::endl;
    //ss >> num_vertices;
    VARSET_NEW(empty, num_vertices);
    edges = std::vector<varset>(num_vertices, empty);
    const_cast<int&>(variableCount) = num_vertices;
    // set all bit to 1
    const_cast<varset&>(all_bit_set) = varset(num_vertices);
    VARSET_SET_ALL(const_cast<varset&>(all_bit_set), num_vertices);
    int row = 0;
    int col = 0;
    do
    {
        std::cout << "row " << row << ", line: " << line << std::endl;
        col = 0; // reset for each line
        tokenizer tok(line, sep);
        for(auto iter=tok.begin(); iter != tok.end(); iter++)
        {
            if(iter->compare("TRUE") == 0 or abs(atof(iter->c_str())) > 0.05 )
            {
                add_edge(row,col);
                std::cout << __FILE__ << ":" << __LINE__ << ", Added edge (" << row+1 << ", " << col+1 << ")\n";
            }
            col++;
        }
        row++;
    }while(std::getline(input_file, line));
    std::cout << "Finished reading MB matrix file, row = " << row << ", col = " << col << std::endl;
    input_file.close();
    const_cast<bool&>(initialized) = true;
    calculate_scc();
    return num_vertices > 0;
}
bool Skeleton::read_file(std::string input_filename)
{
    int num_vertices;
    std::ifstream input_file(input_filename.c_str(), std::ios_base::in);
    if(not input_file.good())
        return false;
    std::string line;
    std::getline(input_file, line);
    std::stringstream ss(line);
    ss >> num_vertices;
    VARSET_NEW(empty, num_vertices);
    edges = std::vector<varset>(num_vertices, empty);
    const_cast<int&>(variableCount) = num_vertices;
    // set all bit to 1
    const_cast<varset&>(all_bit_set) = varset(num_vertices);
    VARSET_SET_ALL(const_cast<varset&>(all_bit_set), num_vertices);
    
    while(std::getline(input_file, line))
    {
        std::stringstream linestream(line);
        int i, j;
        char s; // to absorb comma
        linestream >> i >> s >> j;
        add_edge(i,j);
        std::cout << "Added edge (" << i << ", " << j << ")\n";
    }
    input_file.close();
    const_cast<bool&>(initialized) = true;
    calculate_scc();
    return num_vertices > 0;
}
varset Skeleton::bfs_neighbors(int varIdx, const int max_num_vars_per_cluster, const std::vector<int> & preferred_sequence, const int varIdx2)
{
	VARSET_NEW(cluster, variableCount);
	VARSET_NEW(neighbors, variableCount);

	neighbors = edges[varIdx];
	std::deque<int> mydeck;
	std::queue<int> myq(mydeck);
	// push variable varIdx neighbors into queue
	myq.push(varIdx);
	if(varIdx2 >= 0 and varIdx2 != varIdx)
	    myq.push(varIdx2);
//	for(int i=0; i<variableCount; i++)
////	{
//		int var = preferred_sequence[i];
//		if(VARSET_GET(neighbors, var))
//			myq.push(var);
//	}
	int count = 0; // count the cluster size so far
	while(count < max_num_vars_per_cluster and myq.size())
	{
		int var = myq.front();
		myq.pop();
		VARSET_SET(cluster, var);
		count++;
		// push variable var neighbors
		varset var_neighbors = edges[var];
		for(int i=0; i<variableCount; i++)
		{
			int v = preferred_sequence[i];
			// push the neighbors that are  not in cluster yet
			if(VARSET_GET(var_neighbors, v) and not VARSET_GET(cluster, v))
				myq.push(v);
		}
	}
	return cluster;
}
void Skeleton::bfs_partition(std::vector<varset> & clusters, const int max_num_vars_per_cluster, std::vector<int>  preferred_seq) //partition by Breadth First Search, cluster size limited by max_num_vars_per_cluster
{
	if(preferred_seq.size() == 0)
	{
		preferred_seq.resize(variableCount);
		for(int i=0; i<variableCount; i++)
			preferred_seq[i] = i;
	}
	clusters.clear();
	clusters.reserve(variableCount);
	for(int i=0; i<variableCount; i++)
		clusters.push_back(bfs_neighbors(i, max_num_vars_per_cluster, preferred_seq));
}
int Skeleton::explore_one_scc(int variableIndex, varset & variables_visited, varset & the_scc)
{
    int num = 1;
    VARSET_SET(variables_visited, variableIndex);
    VARSET_SET(the_scc,  variableIndex);
    printf("Just visited variable %d, checking its neighbors\n", variableIndex);
    const varset & neighbors = edges[variableIndex];
    for(int i=0; i<variableCount; i++)
    {
        if(VARSET_GET(variables_visited, i)) // skip if already visited
        {
            continue;
        }
        if(VARSET_GET(neighbors, i))
            num += explore_one_scc(i, variables_visited, the_scc);
    }
    return num;
}
void Skeleton::calculate_scc() // calculate strongly_connected_components
{
    printf("calculate_scc, variableCount = %d\n", variableCount);
    int num_variables_visited = 0;
    VARSET_NEW(variables_visited, variableCount);
    for(int v = 0; v < variableCount and num_variables_visited < variableCount; v++)
    {
        if(VARSET_GET(variables_visited, v)) // if visited already
        {
            printf("calculate_scc, variable %d has been visited\n", v);
            continue;
        }
        // found a new strongly connected component
        VARSET_NEW(the_scc, variableCount);
        int num_variables = explore_one_scc(v, variables_visited, the_scc);
        num_variables_visited += num_variables;
        scc.push_back(the_scc);
        printf("Found strongly connected component with %d variables, scc size,  %d, num_variables_visited %d\n"
              , num_variables, int(scc.size()), num_variables_visited);
        printf("Here are the  components: ");
        for(int i=0; i<variableCount; i++)
            if(VARSET_GET(the_scc,  i))
                printf("%d ", i);
        printf("\n");
    }
}
void Skeleton::print_edges()const
{
    for(size_t i=0; i<edges.size(); i++)
    {
        std::cout << "Edges starting with vertex#  " << i;
        const int maxj = edges.size();
        for(int j=0; j < maxj; j++)
        {
            if(VARSET_GET(edges[i],j))
                std::cout << ","<<j;
//            int newj = VARSET_FIND_NEXT_SET(edges[i], j);
//            if(newj==-1)
//                break;
//            std::cout << ","<< newj;
//            if(newj <=j )
//            {
//                cout << "j not increasing, newj = " << newj
//                     ;
//                j++;
//            }
//            else
//            {
//                j=newj;
//            }
        }
        std::cout << std::endl;
    }
}
//void Skeleton::convert2constraints(Constraints* c)const
//{
//    for(size_t i=0; i <  edges.size(); i++)
//    {
//        varset forbidden = get_forbidden_parents(i);
//        c->expand_forbidden_parents(i, forbidden);
//        std::cout << "Expanded constraints for variable " << i
//                 << ", forbidden " << varsetToString(forbidden)
//                  << std::endl
//                  ;
//    }
//}
}//end namespace datastructures
