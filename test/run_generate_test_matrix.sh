#!/bin/bash
# different matrix widths and heights
n=(100 500 1000 5000 10000 20000 30000 50000)
m=(50 100 500 1000 5000 10000)
p=(3 5 7 10 15 20 50)
for n_i in ${n[*]}
do
  for m_i in ${m[*]}
  do
    # only to ensure that m is not too small
    if (($m_i<$n_i/10)) || (($m_i>$n_i));
    then
      continue
    fi
    for p_i in ${p[*]}
    do
      # only to ensure that p is "reasonable"
      if (($p_i<$m_i/200)) || (($p_i>$m_i/4));
      then
        continue
      fi
      echo "$n_i $m_i $p_i"
      python test/generate_dataset.py -n $n_i -m $m_i -p $p_i -s 13531
    done
  done
done
