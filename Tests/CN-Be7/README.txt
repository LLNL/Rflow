R-matrix modeling of h+a and p+li6 scattering 


R-matriix parameters and data normalizations in h-002_He_004-tt3.xml GNDS file.

Experimental data in test2.data, and initial data normalizations in test2.norms

Previous output files in Outputs/ directory
Intel macbook Pro (2.9 GHz, i7) with Python 3.8 and TF 2.4.0, TFP 0.12.2


Runs:

# Find current chi-squared

../../rflow.py h-002_He_004-tt3.xml test2.data test2.norms | tee h-002_He_004-tt3.xml+test2.out


# Find current chi-squared and plot with data in subdirectory
# Output in h-002_He_004-tt3.xml+test2+C/ directory

../../rflow.py h-002_He_004-tt3.xml test2.data test2.norms -C | tee h-002_He_004-tt3.xml+test2+C.out


# Find current chi-squared, search down to S=1,  and plot with data in subdirectory
# Default steps is I=2000
# Output in h-002_He_004-tt3.xml+test2+S1_I2000+C/ directory
# You can see chi-sq progress on h-002_He_004-tt3.xml+test2+S1_I2000+C/bfgs_min.trace

../../rflow.py h-002_He_004-tt3.xml test2.data test2.norms -C -S 1 | tee h-002_He_004-tt3.xml+test2+C+S1.out

