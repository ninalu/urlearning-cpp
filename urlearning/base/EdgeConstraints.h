#ifndef EDGE_CONSTRAINTS_H
#define EDGE_CONSTRAINTS_H

#include <vector>
#include "typedefs.h"
#include <iostream>

namespace datastructures
{

  class EdgeConstraints
  {
    public:
      EdgeConstraints(int variableCount_)
      : variableCount(variableCount_)
      {
	VARSET_NEW(empty_set, variableCount);
        VARSET_CLEAR_ALL(empty_set);
        direct_parents = std::vector<varset>(variableCount, empty_set) ; 
        ancestors = std::vector<varset>(variableCount, empty_set) ; 
      }
      EdgeConstraints(const EdgeConstraints & rhs)
      : variableCount(rhs.variableCount)
      , direct_parents(rhs.direct_parents.begin(), rhs.direct_parents.end())
      , ancestors(rhs.ancestors.begin(), rhs.ancestors.end())
      {

      }
      varset required_parents(int variable) const
      {
	return direct_parents[variable];
      }
      void remove_one_parent(int variable, int parent_var_idx)
      {
         VARSET_CLEAR(direct_parents[variable], parent_var_idx);	
      }
      bool add_one_parent(int variable, int parent_var_idx)
      {
	/*
	VARSET_NEW(visited, variableCount);
	bool has_cycle = check_cycle(parent_var_idx, variable, visited);
	if(has_cycle)
	{
	  std::cout << "Caused cycle, failed to add parent for variable "<< variable
		    << ", parent " << parent_var_idx
		    ;
	  return false;
	}
	*/
	VARSET_SET(direct_parents[variable], parent_var_idx);
	return true;
      }
      // update ancestors after adding parent_var_idx as parent
      void add_ancestors(int variable, int parent_var_idx)
      {
	VARSET_SET(ancestors[variable], parent_var_idx);
	ancestors[variable] = VARSET_OR(ancestors[variable], ancestors[parent_var_idx]);	
      }
      void addConstraint(int variable, const varset & parents)
      {
	direct_parents[variable] = VARSET_OR(direct_parents[variable], parents);
      }
      bool satisfiesConstraints(int variable, const varset & parents) const
      {
	return VARSET_IS_SUBSET_OF(ancestors[variable], parents);
      }
      bool satisfiesConstraints(int variable, const varset & parents, const varset & cluster_vars) const
      {
	return VARSET_IS_SUBSET_OF(VARSET_AND(ancestors[variable], cluster_vars), parents);
      }
      bool check_cycle(const int start, const int target, varset & visited)
      {
	if(start == target or VARSET_GET(visited, start))
	  return true;
	VARSET_SET(visited, start);
	for(int i=0; i<variableCount; i++)
	{
	  if(VARSET_GET(direct_parents[start], i) and check_cycle(i, target, visited))
	    return true; 
	}
	return false;
      }

    private:
      const int variableCount;
      std::vector<varset> direct_parents;
      std::vector<varset> ancestors;

  };
}//end  namespace datastructures

#endif
