1. The plot finite_size_effect_astar_ges_pc_error_bar6.png was used in the submission. 
   In that plot, PC was run with a wrong parameter and had lower precision. This was fixed in 
   the newer plot finite_size_effect_astar_ges_pc_error_bar7.png. PC performs much better.
   The newer plot will be used in the final paper submission

2. The raw data files are too big to fit within 350 MB, even after zip. 
   So only the first 5 (seqnum 9200 - 9204) data files out of 30 are included in folder raw_data/ .
   More data files will be provided if requested.
   The learned results are in folder result/ , and are summarized in summary/ .

3. The learned DAG/MEC matrices in folder result/ should be interpreted as follows. 

   The ground truth DAGs are named as true_model_xxxx , and their MECs are named as true_model_equiv_xxxx . 
   The xxxx is some 4-digit sequence number such as 9201.
   For true_model_xxxx, if the entry (i,j) is 1, then it means directed edge  j -> i
   The MEC is in file true_model_equiv_xxxx. The edge interpretation is opposite. 
   If the (i,j) entry is 1, then it means edge i -> j . 
   If both entries (i,j) and (j,i) are 1, then undirected edge i-j

   For GES, tges1_N10000_9200.csv is the GES learned MEC matrix for raw_data_9200.csv, lambda=1, sample size 10000. 
   If the (i,j) entry is 1, then it means edge i -> j . 
   If both entries (i,j) and (j,i) are 1, then undirected edge i-j .

   The GES DAG files are labeled with _dag_ in file names, such as tges1_N10000_dag_9200.csv
   If the (i,j) entry is 1, then it means edge i -> j .
   This is opposite to the ground truth DAG matrix. 
   It is the result I get directly from pcalg, although I probably should have transposed it. 
   
   The A* learned files follow astar${lambda}_N${Nsize}_${seqnum}.csv
   It is a DAG. If the entry (i,j) is 1, then it means directed edge  j -> i
   
   The NOTEARS learned files follow notears${lambda}_N{Nsize}_${seqnum}.csv .
