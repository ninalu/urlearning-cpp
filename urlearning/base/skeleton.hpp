/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Skeleton.hpp
 * Author: nilu
 *
 * Created on April 23, 2017, 3:47 PM
 */

#ifndef SKELETON_HPP
#define SKELETON_HPP

#define NATIVE_VARSET

#include "urlearning/base/typedefs.h"
#include <vector>   
namespace datastructures {
    
// forward declaration
//class Constraints;

class Skeleton
{
public:
    // default constructor, need at least one variable
    Skeleton(int variableCount_ = 1):initialized(false),variableCount(variableCount_),all_bit_set(varset(variableCount_))
    {
        // set all bit to 1
        VARSET_SET_ALL(const_cast<varset&>(all_bit_set), variableCount);
    }
    Skeleton(std::string input_filename);
    bool read_file(std::string input_filename);
    bool read_matrix_file(std::string input_filename);
    bool read_arc_list_file(std::string input_filename, int num_variables);
    void add_edge(int i, int j) // add edge (i,j)
    {
        VARSET_SET(edges[i], j);
        VARSET_SET(edges[j], i);
    }
    void remove_edge(int i, int j)
    {
        VARSET_CLEAR(edges[i],j);
        VARSET_CLEAR(edges[j],i);
    }
    bool edge_exists(int i, int j)const
    {
        return VARSET_GET(edges[i],j);
    }
    const varset  get_forbidden_parents(int varIdx)const
    {
        return VARSET_NOT(edges[varIdx]); // temporary solution, more efficient if using edges[varIdx] directly
    }
    const varset & get_neighbors(int varIdx)const
    {
        return initialized ? edges[varIdx] : all_bit_set;
    }
    void set_variable_count(int num_vertices)
    {
        const_cast<int&>(variableCount) = num_vertices;
        // set all bit to 1
        const_cast<varset&>(all_bit_set) = varset(num_vertices);
        VARSET_SET_ALL(const_cast<varset&>(all_bit_set), num_vertices);
        one_scc_only.push_back(all_bit_set);
    }
    int size()const {return variableCount;}
    bool good()const{return initialized;}
//    void convert2constraints(Constraints* c)const;
    void calculate_scc(); // calculate strongly_connected_components
    const std::vector<varset> & get_scc() const  {return good() ? scc : one_scc_only;} // retrieve  strongly connected components
    void print_edges()const;
    varset bfs_neighbors(int varIdx, const int max_num_vars_per_cluster, const std::vector<int> & preferred_sequence, const int varIdx2 = -1 ); //partition by Breadth First Search, cluster size limited by max_num_vars_per_cluster
    void bfs_partition(std::vector<varset> & clusters, const int max_num_vars_per_cluster, std::vector<int> preferred_sequence); //partition by Breadth First Search, cluster size limited by max_num_vars_per_cluster

    
private:
    int explore_one_scc(int variableIndex, varset & variables_visited, varset & the_scc);
    const bool initialized;   // Indicate whether this skeleton has been initialized
    const int variableCount;  // number of variables
    const varset all_bit_set; // bit map  where all bits set as 1, to be used when skeleton is not initialized so that it won't filter anything
    std::vector<varset> one_scc_only;
    std::vector<varset> edges;
    std::vector<varset> scc;
};
    

}//end namespace datastructures
#endif /* SKELETON_HPP */

