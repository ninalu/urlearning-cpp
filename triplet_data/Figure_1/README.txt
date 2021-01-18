The learned DAG/MEC matrices should be interpreted as follows. 
Files are named after the methods. For example, pc_mec_8000.csv is the learned MEC by PC algorithm.
If the file name is labeled as _dag_ , then it should be interpretted as DAG.
If the entry (i,j) is 1, then it means directed edge  j -> i

If the file name is labeled as _mec_ , the it should be interpretted as MEC.
   If the (i,j) entry is 1, then it means edge i -> j . 
   If both entries (i,j) and (j,i) are 1, then undirected edge i-j .

