# 
# Ian Thompson.  Nov 20, 2020
#
# The N15 here is the compound system.
# EXFOR uses N-15 to start with, then GNDS uses N15

# clean previous
rm -rf Data_X4
#
# get X4 files into *.dat format, as used in IAEA project
# check csv file for correctness. 
# delete (or move out of this directory) any unwanted X4 datasets
do.getX4 N-15 0.01
# 
# This expands to
../getX4cn4data.py N-15 0.01 | tee getX4cn-N15-above0.01@.out
#
# Read all *.dat files and csv files, to put into big files flow.data and flow.norms
do.dataflow Data_X4/*.dat

# This expands to:
../data4rflow.py -d Data_X4 -p datafile.props.csv n 20. Data_X4/*.dat --pops pops-global.xml N15-pops.xml |& tee data4rflow-Data_X4.out
#
# Make a trial R-matrix parameter set (e.g. trial.xml) SOMEHOW!!
#
# See how the first data fits from the *.json files produced by :
#
# ../rflow.py trial.xml flow.data flow.norms -MC
