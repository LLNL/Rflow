# 
# Ian Thompson. Jan 9, 2021
#
# The N15 here is the compound system.

# clean previous
rm -rf Data_X4
#
# get all X4 files into *.dat format, as used in IAEA project, and csv file for information. Include N-nat as N-14
../getX4cn4data.py N15 -n N-14 | tee getX4cn-N15n.out
#
# check csv file for correctness. 
# delete (or move out of this directory) any unwanted X4 datasets
#
# Change norm of Mani_F0428002 to 2.44e-5
#
# Read all *.dat files and csv files, to put into big files flow.data and flow.norms
../data4rflow.py -d Data_X4 --CSV datafile.props.csv --Projectiles  n H1 He4 -L 1 0 1 -o flow-n1p0a1.data -n flow-n1p0a1.norms --EminCN 7.0  --EmaxCN 20. -E 20.0 --pops  ../ripl2pops_to_Z8.xml -j 2.5 -p 1 -w 0.1  -I Data_X4/*.dat -T 1 |& tee data4rflow-Data_X4-B10C20.out

# shorter:
../data4rflow.py -P n H1 He4 -L 1 0 1 -o flow-n1p0a1.data -n flow-n1p0a1.norms --EminCN 7.0  --EmaxCN 20. -j 2.5 -p 1 -w 0.1  -I Data_X4/*.dat |& tee n1p0a1-data4rflow-Data_X4-B7C20.out
../data4rflow.py -P He4 n H1 -L 1 1 0 -o flow-a1n1p0.data -n flow-a1n1p0.norms --EminCN 7.0  --EmaxCN 20. -j 2.5 -p 1 -w 0.1  -I Data_X4/*.dat |& tee a1n1p0-data4rflow-Data_X4-B7C20.out
../data4rflow.py -P He4 n H1 -L 2 2 1 -o flow-a2n2p1.data -n flow-a2n2p1.norms --EminCN 7.0  --EmaxCN 20. -j 2.5 -p 1 -w 0.1  -I Data_X4/*.dat |& tee a2n2p1-data4rflow-Data_X4-B7C20.out


# Maybe summarize as ##../do.dataflow Data_X4/*.dat -E 10.0

#
# Make a trial R-matrix parameter set (e.g. N15r.sfresco-a+.xml) by     
ferdinand.py N15r.sfresco -a xml
#
# See how the first data fits from the *.json files produced by :
#
../rflow.py N15r.sfresco-a+.xml flow.data flow.norms -M |& tee N15r-sfresco-rflow-1.out

../rflow.py Fitted-gnds1.xml flow-npa.data flow-npa.norms -MC -a 0 |& tee  Fitted-gnds1.xml+flow-npa_a0.out

../rflowt.py Fitted-gnds1.xml flow-npa.data flow-npa.norms -T -a 0  -MCT -a 0 -t v | tee N15r.sfresco-a+.xml+flow-npa-a0-v.out
