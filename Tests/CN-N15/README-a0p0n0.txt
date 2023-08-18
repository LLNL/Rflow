# 
# Ian Thompson. Jan 25, 2022
#
# The N15 here is the compound system.

# clean previous
rm -rf Data_X4s
#
# get all X4s files into *.dat format, as used in IAEA project, and csv file for information. Include N-nat as N-14
../../getX4cn4datas.py N15 -n N-14 | tee getX4cn-N15n.out
#
# check csv file for correctness. 
# delete (or move out of this directory) any unwanted X4s datasets
#
#
# Read all *.dat files and csv files, to put into big files flow.data and flow.norms for rflow.py
#

# A0P0N0
#
../../data4rflows.py -d Data_X4s --CSV datafile.props.csv --Projectiles  He4 H1 n -L 0 0 0 -o flow-a0p0n0.data -n flow-a0p0n0.norms --EminCN 7.0  --EmaxCN 20. -E 20.0 --pops  ../ripl2pops_to_Z8.xml -j 2.5 -p 1 -w 0.1 -I Data_X4s/*.dat -T 1 |& tee data4rflow-Data_X4s-a0p0n0-B10C20.out
#
# no cross-sections (-a 0)
../../rflow.py N15r-a0p0n0-a+.xml flow-a0p0n0.data flow-a0p0n0.norms  --Cross_Sections --TransitionMatrix 1  --tag v --anglesData 0 --EMAX 10 | tee N15r-a0p0n0-a+.xml+flow-a0p0n0-E10.0_a0+C_v.out

# plot results for individual EXFOR sets
../plotjson N15r-a0p0n0-a+.xml+flow-a0p0n0-E10.0_a0+C_v/*@*json
# plot results for excitation functions
../plotjson N15r-a0p0n0-a+.xml+flow-a0p0n0-E10.0_a0+C_v/Angle-integrals-*json'


# no cross-sections (-a 0), and excluding Harvey data (4135 points) for speed.
../../rflow.py N15r-a0p0n0-a+.xml flow-a0p0n0.data flow-a0p0n0.norms  --Cross_Sections --TransitionMatrix 1  --tag xHarvey --anglesData 0 --EMAX 10 -x Harvey | tee N15r-a0p0n0-a+.xml+flow-a0p0n0-E10.0_a0+C_xHarvey.out


# Up to 4 MeV max, no cross-sections (-a 0), and excluding Harvey data (4135 points) for speed.
../../rflow.py N15r-a0p0n0-a+.xml flow-a0p0n0.data flow-a0p0n0.norms  --Cross_Sections --TransitionMatrix 1  --tag xHarvey --anglesData 0 --EMAX 4 -x Harvey | tee N15r-a0p0n0-a+.xml+flow-a0p0n0-E4.0_a0+C_xHarvey.out
