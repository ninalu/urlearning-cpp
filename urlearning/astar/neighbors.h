#include <stdint.h>
template<int NumVars>
struct Neighbors
{
  static const int num_vars = NumVars;
  uint8_t children[NumVars];  // children, vc + oc
  uint8_t nb[NumVars];  // astar neighbors 
  uint8_t vp[NumVars];  // v-structure parents
  uint8_t vc[NumVars];  // v-structure children
  uint8_t op[NumVars];  // orientation rule parents
  uint8_t oc[NumVars];  // orientation rule children
  uint8_t num_nb; // num of neighbors
  uint8_t num_vp; // num of v-structure parents
  uint8_t num_vc; // num of v-structure children
  uint8_t num_op; // num of orientation rule parents
  uint8_t num_oc; // num of orientation rule children
  uint8_t num_ch; // number of children

  // default constructor
  Neighbors():num_nb(0),num_vp(0), num_vc(0),num_op(0), num_oc(0), num_ch(0){}
  int add_neighbor(const int j)
  {
    // check duplicate
    bool found = false;
    for(int i=0; i<num_nb; i++)
    {
      if(j == nb[i]) 
      {
	found = true;
	break;
      }
    }
    if(not found)
      nb[num_nb++] = j;
    return num_nb;
  }

  // add v-structure parent
  int add_vp(const int j)
  {
    // check duplicate
    bool found = false;
    for(int i=0; i<num_vp; i++)
    {
      if(j == vp[i]) 
      {
	found = true;
	break;
      }
    }
    if(not found)
      vp[num_vp++] = j;
    // add into neighbors
    add_neighbor(j);
    return num_vp;
  }

  // add child
  int add_child(const int j)
  {
     bool found = false;
     for(int i=0; i<num_ch; i++)
     {
	if(j == children[i])
	{
	  found = true;
	  break;
	}
     }
     if(not found)
       children[num_ch++] = j;
     return num_ch;
  }

  // add v-structure child
  int add_vc(const int j)
  {
    // check duplicate
    bool found = false;
    for(int i=0; i<num_vc; i++)
    {
      if(j == vc[i]) 
      {
	found = true;
	break;
      }
    }
    if(not found)
      vc[num_vc++] = j;
    // add into neighbors
    add_neighbor(j);
    add_child(j);
    return num_vc;
  }

  // add orientation rule parent
  int add_op(const int j)
  {
    // check duplicate
    bool found = false;
    for(int i=0; i<num_op; i++)
    {
      if(j == op[i]) 
      {
	found = true;
	break;
      }
    }
    if(not found)
      op[num_op++] = j;
    add_neighbor(j);
    return num_op;
  }

  // add orientation child
  int add_oc(const int j)
  {
    // check duplicate
    bool found = false;
    for(int i=0; i<num_oc; i++)
    {
      if(j == oc[i]) 
      {
	found = true;
	break;
      }
    }
    if(not found)
      oc[num_oc++] = j;
    add_child(j);
    add_neighbor(j);
    return num_oc;
  }
};


