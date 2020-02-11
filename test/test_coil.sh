#!/bin/bash

# Run GNMF on COIL20 dataset given different lambda, number of neighbors, and iterations
#   Number of object            = 15
#   Number of images per object = 10
# different parameters
lmbda=(0.005 0.01 0.02 0.05 0.08 0.1)
neighbor=(5 6 7)
iterations=(200 500)
for l in ${lmbda[*]}
do
  for p in ${neighbor[*]}
  do
    for iter in ${iterations[*]}
    do
      python dataset/COIL20/run.py -l $l -p $p -i $iter -k 15 -n 10
    done
  done
done
