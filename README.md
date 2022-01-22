Rflow

R-matrix fitting EXFOR data using tensorflow

usage: rflow.py [-h] [-x [EXCL [EXCL ...]]] [-1] [-F [FIXED [FIXED ...]]] [-n]
                [-r RESTARTS] [-B BACKGROUND] [--BG] [-R] [--LMatrix]
                [--groupAngles GROUPANGLES] [-a ANGLESDATA] [-m MAXDATA]
                [-e EMIN] [-E EMAX] [-p PMIN] [-P PMAX] [-d DMIN] [-D DMAX]
                [-N NLMAX] [-L LAMBDA] [--ABES] [-G GRID] [-S SEARCH]
                [-I ITERATIONS] [-i INIT INIT] [-A AVERAGING] [-w WIDTHWEIGHT]
                [-X XCLUDE] [--Large LARGE] [-C] [-c] [-T TRANSITIONMATRIX]
                [-s] [-M MULTI] [--ML ML] [--datasize size] [-l LOGS] [-t TAG]
                [-v] [-g]
                inFile dataFile normFile

Compare R-matrix Cross sections with Data

positional arguments:
  inFile                The intial gnds R-matrix set
  dataFile              Experimental data to fit
  normFile              Experimental norms for fitting

optional arguments:
  -h, --help            show this help message and exit
  -x [EXCL [EXCL ...]], --exclude [EXCL [EXCL ...]]
                        Substrings to exclude if any string within group name
  -1, --norm1           Start with all norms=1
  -F [FIXED [FIXED ...]], --Fixed [FIXED [FIXED ...]]
                        Names of variables (as regex) to keep fixed in
                        searches
  -n, --normsfixed      Fix all physical experimental norms (but not free
                        norms)
  -r RESTARTS, --restarts RESTARTS
                        max restarts for search
  -B BACKGROUND, --Background BACKGROUND
                        Pole energy (lab) above which are all distant poles.
                        Fixed in searches.
  --BG                  Include BG in name of background poles
  -R, --ReichMoore      Include Reich-Moore damping widths in search
  --LMatrix             Use level matrix method if not already Brune basis
  --groupAngles GROUPANGLES
                        Unused. Number of energy batches for T2B transforms,
                        aka batches
  -a ANGLESDATA, --anglesData ANGLESDATA
                        Max number of angular data points to use (to make
                        smaller search). Pos: random selection. Neg: first
                        block
  -m MAXDATA, --maxData MAXDATA
                        Max number of data points to use (to make smaller
                        search). Pos: random selection. Neg: first block
  -e EMIN, --emin EMIN  Min cm energy for gnds projectile.
  -E EMAX, --EMAX EMAX  Max cm energy for gnds projectile.
  -p PMIN, --pmin PMIN  Min energy of R-matrix pole to fit, in gnds cm energy
                        frame. Overrides --Fixed.
  -P PMAX, --PMAX PMAX  Max energy of R-matrix pole to fit. If p>P, create
                        gap.
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
                        Add widthWeight*vary_widths**2 to chisq during
                        searches
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
  -M MULTI, --Multi MULTI
                        Which Mirrored Strategy in TF
  --ML ML               MLcompute device for Macs
  --datasize size       Font size for experiment symbols. Default=0.2
  -l LOGS, --logs LOGS  none, x, y or xy for plots
  -t TAG, --tag TAG     Tag identifier for this run
  -v, --verbose         Verbose output
  -g, --debug           Debugging output (more than verbose)
