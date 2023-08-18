# 
# Ian Thompson. June 5, 2023
#
# The N15 here is the compound system.

# clean previous
rm -rf Data_X4s
#
# get all X4s files into *.dat format, as used in IAEA project, and csv file for information. Include N-nat as N-14
../../getX4cn4datas.py N15 -n N-14 | tee getX4cn-N15n.out
#
# check  Data_X4s/datafile.props.csv file for correctness. 
# delete (or move out of this directory) any unwanted X4s datasets
#
#
# Read all *.dat files and csv files, to put into big files flow.data and flow.norms for rflow.py
#
# A1P0N1
012


# Calculate cross-sections and transitions with existing R-matrix parameters (no searching)

# no angular cross-sections (-a 0) up to 10 MeV
../../rflow.py N15r-a1p0n1-a+.xml flow-a1p0n1.data flow-a1p0n1.norms  --Cross_Sections --TransitionMatrix 1  --tag v --anglesData 0 --EMAX 10 | tee N15r-a1p0n1-a+.xml+flow-a1p0n1-E10.0_a0+C_v.out

# plot results for individual EXFOR sets
../plotjson N15r-a1p0n1-a+.xml+flow-a1p0n1-E10.0_a0+C_v/*@*json

# plot results for excitation functions
../plotjson N15r-a1p0n1-a+.xml+flow-a1p0n1-E10.0_a0+C_v/Angle-integrals-*json'


# Search excluding Harvey Dayras Lee_Jr data for speed, and Sanders Mani-F0465002 Liu-C0887002 data for normalization uncertainties. Up to 10 MeV
# The results of this search are included in the code release, parameters in N15r-a1p0n1-a+b3.xml+flow-a1p0n1-E10.0+S10_I300+C_xHDLCSM-fit.xml 
#
../../rflow.py N15r-a1p0n1-a+b3.xml flow-a1p0n1.data flow-a1p0n1.norms --Cross_Sections --TransitionMatrix 1 --EMAX 10 -t xHDLCSM -x Harvey Dayras Lee_Jr Sanders Mani-F0465002 Liu-C0887002 -S 10 -I 300 | tee N15r-a1p0n1-a+b3.xml+flow-a1p0n1-E10.0+S10_I300+C_xHDLCSM.out



