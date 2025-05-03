# Rflow:  R-matrix methods for fitting EXFOR data using tensorflow
	 Version 0.30
	 Release: LLNL-CODE-853144
	 SPDX-License-Identifier: MIT
###  Ian J. Thompson

	 Email: thompson97@llnl.gov
	   
## Needed libraries

Users to download:

**fudge** version > 6 from [github.com/LLNL/fudge](https://github.com/LLNL/fudge),
  for example the tag at [github.com/LLNL/fudge/releases/tag/6.1.0](https://github.com/LLNL/fudge/releases/tag/6.1.0). Include fudge in PYTHONPATH.


**yaml** files: install libyaml, py-yaml

**tensorflow** from [www.tensorflow.org/install](https://www.tensorflow.org/install).
For macs optionally see [developer.apple.com/metal/tensorflow-plugin/](https://developer.apple.com/metal/tensorflow-plugin/)

For getX4cn4datas.py: 
**x4i** from [github.com/brown170/x4i.git](https://github.com/brown170/x4i.git)


## Example conda installation (May 2025)
 
	conda create --name T2
	conda activate T2 

	conda install -c apple tensorflow-deps # get 2.9.0, python 3.10.16, numpy 1.22.3
	python -m pip install tensorflow # get 2.19.0, python 3.10.16, numpy 2.1.3
	python -m pip install tensorflow_probability # get 0.25
	python -m pip install tensorflow_keras # get 2.19. Needed for tensorflow_probability
	python -m pip install pyyaml # get 6.0.2
	python -m pip install scipy # get 1.15.2. Needed for loggamma

	python -m pip install tensorflow-metal # get 1.2.0, for mac GPU (optional: the CPU calculations use all cpus well)

	(the # get comments summarise my results, and should not be pasted as part of the command!)

## R-matrix fitting EXFOR data using tensorflow
```
usage: rflow.py [-h] [-x [EXCL ...]] [--ExcludeFile EXCLUDEFILE] [-1]
                [-F [FIXED ...]] [--FixedFile FIXEDFILE] [-n]
                [--nonzero NONZERO] [-r RESTARTS] [-B BACKGROUND]
                [--BG] [-R] [--LMatrix] [--groupAngles GROUPANGLES]
                [-a ANGLESDATA] [-m MAXDATA] [-e EMIN] [-E EMAX]
                [-p PMIN] [-P PMAX] [-d DMIN] [-D DMAX] [-N NLMAX]
                [-L LAMBDA] [--ABES] [-G GRID] [-S SEARCH]
                [-I ITERATIONS] [-i INIT INIT] [-A AVERAGING]
                [-w WIDTHWEIGHT] [-X XCLUDE] [--Large LARGE] [-C] [-c]
                [-T TRANSITIONMATRIX] [-s] [-M MULTI] [--datasize size]
                [-l LOGS] [-t TAG] [-v] [-g]
                inFile dataFile normFile
```
## Compare R-matrix Cross sections with Data

### positional arguments:
	  inFile                The intial gnds R-matrix set
	  dataFile              Experimental data to fit
	  normFile              Experimental norms for fitting

### optional arguments:
```
  -h, --help            show this help message and exit
  -x [EXCL [EXCL ...]], --exclude [EXCL [EXCL ...]]
                        Substrings to exclude if any string within group name
  -1, --norm1           Start with all norms=1
  -F [FIXED [FIXED ...]], --Fixed [FIXED [FIXED ...]]
                        Names of variables (as regex) to keep fixed in searches
  --ExcludeFile EXCLUDEFILE
                        Name of file with names of variables (as regex) to
                        exclude if any string within group name
  -n, --normsfixed      Fix all physical experimental norms (but not free norms)
  -r RESTARTS, --restarts RESTARTS
                        max restarts for search
  -B BACKGROUND, --Background BACKGROUND
                        Pole energy (lab) above which are all distant poles.
                        Fixed in searches.
  --BG                  Include BG in name of background poles
  -R, --ReichMoore      Include Reich-Moore damping widths in search
  --LMatrix             Use level matrix method if not already Brune basis
  --groupAngles GROUPANGLES
                        Unused. Number of energy batches for T2B transforms, aka
                        batches
  -a ANGLESDATA, --anglesData ANGLESDATA
                        Max number of angular data points to use (to make
                        smaller search). Pos: random selection. Neg: first block
  -m MAXDATA, --maxData MAXDATA
                        Max number of data points to use (to make smaller
                        search). Pos: random selection. Neg: first block
  -e EMIN, --emin EMIN  Min cm energy for gnds projectile.
  -E EMAX, --EMAX EMAX  Max cm energy for gnds projectile.
  -p PMIN, --pmin PMIN  Min energy of R-matrix pole to fit, in gnds cm energy
                        frame. Overrides --Fixed.
  -P PMAX, --PMAX PMAX  Max energy of R-matrix pole to fit. If p>P, create gap.
  -d DMIN, --dmin DMIN  Min energy of R-matrix pole to fit damping, in gnds cm
                        energy frame.
  -D DMAX, --DMAX DMAX  Max energy of R-matrix pole to fit damping. If d>D,
                        create gap.
  -N NLMAX, --NLMAX NLMAX
                        Max number of partial waves in one reaction pair
  -L LAMBDA, --Lambda LAMBDA
                        Use (E-dmin)^Lambda to modulate all damping widths at
                        gnds-scattering cm energy E.
  --ABES                Allow Brune Energy Shifts. Use inexact method
  -G GRID, --Grid GRID  Make energy grid with this energy spacing (MeV) for 1d
                        interpolation, default 0.001
  -S SEARCH, --Search SEARCH
                        Search minimization target.
  -I ITERATIONS, --Iterations ITERATIONS
                        max_iterations for search
  -i INIT INIT, --init INIT INIT
                        iterations and snap file name for starting parameters
  -A AVERAGING, --Averaging AVERAGING
                        Averaging width to all scattering: imaginary =
                        Average/2.
  -w WIDTHWEIGHT, --widthWeight WIDTHWEIGHT
                        Add widthWeight*vary_widths**4 to chisq during searches
  -X XCLUDE, --XCLUDE XCLUDE
                        Make dataset*3 with data chi < X (e.g. X=3). Needs -C
                        data.
  --Large LARGE         'large' threshold for parameter progress plotts.
  -C, --Cross_Sections  Output fit and data files, for json and grace
  -c, --compound        Plot -M and -C energies on scale of E* of compound
                        system
  -T TRANSITIONMATRIX, --TransitionMatrix TRANSITIONMATRIX
                        Produce cross-section transition matrix functions in
                        *tot_a and *fch_a-to-b
  -s, --single          Single precision: float32, complex64
  --datasize size       Font size for experiment symbols. Default=0.2
  -l LOGS, --logs LOGS  none, x, y or xy for plots
  -t TAG, --tag TAG     Tag identifier for this run
  -v, --verbose         Verbose output
  -g, --debug           Debugging output (more than verbose)
```

## Data preparation codes
### getX4cn4datas.py
```
	Extract EXFOR data using x4i calls.

usage: getX4cn4datas.py [-h] [-E ENERGYMAX] [-e ENERGYMIN]
                        [-p PROJECTILES] [-n NAT] [-d DIR]
                        [-i [INCL ...]] [-x [EXCL ...]] [--pops POPS]
                        [--allowNeg] [-t TOLERANCE]
                        CN

```
### data4rflows.py
	Prepare data for Rflow
```	
usage: data4rflows.py [-h] [-P PROJECTILES [PROJECTILES ...]]
                      [-L LEVELSMAX [LEVELSMAX ...]] [-B EMINCN]
                      [-C EMAXCN] [-J JMAX] [-e EMINP] [-E EMAXP]
                      [-r RMATRIX_RADIUS] [-G] [-R] [-j JDEF]
                      [-p PIDEF] [-w WIDEF] [-F]
                      [-I INFILES [INFILES ...]] [-s SCALEFACTORS]
                      [--pops POPS] [--pops2 POPS2] [-d DIR] [-o OUT]
                      [-n NORMS] [-S] [--SF] [-T TERM0] [-M MAXPARS]
                      [--CSV CSV] [-a ADJUSTS] [-f FITS]
```
See README files in subfolders of Tests for examples of using these codes.

## Other standalone codes
```
descending.py:		order chisq trace data in descending order
endf2flow.py:		make Rflow input data directly from an ENDF evaluation
gnds_merge.py: 		merge parts of two R-matrix parameter sets
gnds_mod.py:		modify an R-matrix parameter set
plotLevels.py:		plot compound nucleus levels from R-matrix levels
pops-read.py:		list numbers of nuclear levels in a PoPs file
sketchChannels.py:	plot levels for multiple mass partitions of a nuclide.
snapnorm.py:		plot largest search parameters from a snap file.
romp.py				make a R-matrix parameter set from YAHFC level densities
						 and optical potentials.
levels2sfresco.py	read ENSDF levels from RIPL xml file and make sfresco list of resonances
```
