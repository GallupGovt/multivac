#!/usr/bin/env bash

for i in {1..5}
do
  traintimes=$((500*$i))
  learning_rate=$(expr 0.001*$i | bc)
  number_batches=$((50*$i))

  python3 src/rdf_graph/map_queries.py -d data/ -m transh -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/transh_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt &
  python3 src/rdf_graph/map_queries.py -d data/ -m analogy -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/analogy_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt &
  python3 src/rdf_graph/map_queries.py -d data/ -m complex -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/complex_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt &
  python3 src/rdf_graph/map_queries.py -d data/ -m distmult -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/distmult_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt &
  python3 src/rdf_graph/map_queries.py -d data/ -m hole -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/hole_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt &
  python3 src/rdf_graph/map_queries.py -d data/ -m rescal -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/rescal_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt &
  python3 src/rdf_graph/map_queries.py -d data/ -m transd -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/transd_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt &
  python3 src/rdf_graph/map_queries.py -d data/ -m transe -o output/ -g glove/ -s sir -r fit -j $traintimes -a $learning_rate -b $number_batches > openke_testing/transe_tt${traintimes}_lr${learning_rate}_nb$number_batches.txt 
done
