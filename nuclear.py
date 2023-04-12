# <<BEGIN-copyright>>
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LLNL Nuclear Data and Theory group
#         (email: mattoon1@llnl.gov)
# LLNL-CODE-683960.
# All rights reserved.
# 
# This file is part of the FUDGE package (For Updating Data and 
#         Generating Evaluations)
# 
# When citing FUDGE, please use the following reference:
#   C.M. Mattoon, B.R. Beck, N.R. Patel, N.C. Summers, G.W. Hedstrom, D.A. Brown, "Generalized Nuclear Data: A New Structure (with Supporting Infrastructure) for Handling Nuclear Data", Nuclear Data Sheets, Volume 113, Issue 12, December 2012, Pages 3145-3171, ISSN 0090-3752, http://dx.doi.org/10. 1016/j.nds.2012.11.008
# 
# 
#     Please also read this link - Our Notice and Modified BSD License
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the disclaimer below.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the disclaimer (as noted below) in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of LLNS/LLNL nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific
#       prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
# THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# 
# Additional BSD Notice
# 
# 1. This notice is required to be provided under our contract with the U.S.
# Department of Energy (DOE). This work was produced at Lawrence Livermore
# National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
# 
# 2. Neither the United States Government nor Lawrence Livermore National Security,
# LLC nor any of their employees, makes any warranty, express or implied, or assumes
# any liability or responsibility for the accuracy, completeness, or usefulness of any
# information, apparatus, product, or process disclosed, or represents that its use
# would not infringe privately-owned rights.
# 
# 3. Also, reference herein to any specific commercial products, process, or services
# by trade name, trademark, manufacturer or otherwise does not necessarily constitute
# or imply its endorsement, recommendation, or favoring by the United States Government
# or Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
# herein do not necessarily state or reflect those of the United States Government or
# Lawrence Livermore National Security, LLC, and shall not be used for advertising or
# product endorsement purposes.
# 
# <<END-copyright>>

elementsZSymbolName = {
      0 : ( "n",  "Neutron" ),        1 : ( "H",  "Hydrogen" ),       2 : ( "He", "Helium" ),         3 : ( "Li", "Lithium" ),       4 : ( "Be", "Beryllium" ), 
      5 : ( "B",  "Boron" ),          6 : ( "C",  "Carbon" ),         7 : ( "N",  "Nitrogen" ),       8 : ( "O",  "Oxygen" ),        9 : ( "F",  "Fluorine" ), 
     10 : ( "Ne", "Neon" ),          11 : ( "Na", "Sodium" ),        12 : ( "Mg", "Magnesium" ),     13 : ( "Al", "Aluminium" ),    14 : ( "Si", "Silicon" ), 
     15 : ( "P",  "Phosphorus" ),    16 : ( "S",  "Sulphur" ),       17 : ( "Cl", "Chlorine" ),      18 : ( "Ar", "Argon" ),        19 : ( "K",  "Potassium" ),
     20 : ( "Ca", "Calcium" ),       21 : ( "Sc", "Scandium" ),      22 : ( "Ti", "Titanium" ),      23 : ( "V",  "Vanadium" ),     24 : ( "Cr", "Chromium" ), 
     25 : ( "Mn", "Manganese" ),     26 : ( "Fe", "Iron" ),          27 : ( "Co", "Cobalt" ),        28 : ( "Ni", "Nickel" ),       29 : ( "Cu", "Copper" ), 
     30 : ( "Zn", "Zinc" ),          31 : ( "Ga", "Gallium" ),       32 : ( "Ge", "Germanium" ),     33 : ( "As", "Arsenic" ),      34 : ( "Se", "Selenium" ), 
     35 : ( "Br", "Bromine" ),       36 : ( "Kr", "Krypton" ),       37 : ( "Rb", "Rubidium" ),      38 : ( "Sr", "Strontium" ),    39 : ( "Y",  "Yttrium" ),
     40 : ( "Zr", "Zirconium" ),     41 : ( "Nb", "Niobium" ),       42 : ( "Mo", "Molybdenum" ),    43 : ( "Tc", "Technetium" ),   44 : ( "Ru", "Ruthenium" ), 
     45 : ( "Rh", "Rhodium" ),       46 : ( "Pd", "Palladium" ),     47 : ( "Ag", "Silver" ),        48 : ( "Cd", "Cadmium" ),      49 : ( "In", "Indium" ), 
     50 : ( "Sn", "Tin" ),           51 : ( "Sb", "Antimony" ),      52 : ( "Te", "Tellurium" ),     53 : ( "I",  "Iodine" ),       54 : ( "Xe", "Xenon" ), 
     55 : ( "Cs", "Cesium" ),        56 : ( "Ba", "Barium" ),        57 : ( "La", "Lanthanum" ),     58 : ( "Ce", "Cerium" ),       59 : ( "Pr", "Praseodymium" ),
     60 : ( "Nd", "Neodymium" ),     61 : ( "Pm", "Promethium" ),    62 : ( "Sm", "Samarium" ),      63 : ( "Eu", "Europium" ),     64 : ( "Gd", "Gadolinium" ), 
     65 : ( "Tb", "Terbium" ),       66 : ( "Dy", "Dysprosium" ),    67 : ( "Ho", "Holmium" ),       68 : ( "Er", "Erbium" ),       69 : ( "Tm", "Thulium" ), 
     70 : ( "Yb", "Ytterbium" ),     71 : ( "Lu", "Lutetium" ),      72 : ( "Hf", "Hafnium" ),       73 : ( "Ta", "Tantalum" ),     74 : ( "W",  "Tungsten" ), 
     75 : ( "Re", "Rhenium" ),       76 : ( "Os", "Osmium" ),        77 : ( "Ir", "Iridium" ),       78 : ( "Pt", "Platinum" ),     79 : ( "Au", "Gold" ),
     80 : ( "Hg", "Mercury" ),       81 : ( "Tl", "Thallium" ),      82 : ( "Pb", "Lead" ),          83 : ( "Bi", "Bismuth" ),      84 : ( "Po", "Polonium" ), 
     85 : ( "At", "Astatine" ),      86 : ( "Rn", "Radon" ),         87 : ( "Fr", "Francium" ),      88 : ( "Ra", "Radium" ),       89 : ( "Ac", "Actinium" ), 
     90 : ( "Th", "Thorium" ),       91 : ( "Pa", "Protactinium" ),  92 : ( "U",  "Uranium" ),       93 : ( "Np", "Neptunium" ),    94 : ( "Pu", "Plutonium" ), 
     95 : ( "Am", "Americium" ),     96 : ( "Cm", "Curium" ),        97 : ( "Bk", "Berkelium" ),     98 : ( "Cf", "Californium" ),  99 : ( "Es", "Einsteinium" ),
    100 : ( "Fm", "Fermium" ),      101 : ( "Md", "Mendelevium" ),  102 : ( "No", "Nobelium" ),     103 : ( "Lr", "Lawrencium" ),  104 : ( "Rf", "Rutherfordium" ), 
    105 : ( "Db", "Dubnium" ),      106 : ( "Sg", "Seaborgium" ),   107 : ( "Bh", "Bohrium" ),      108 : ( "Hs", "Hassium" ),     109 : ( "Mt", "Meitnerium" ), 
    110 : ( "Ds", "Darmstadtium" ), 111 : ( "Rg", "Roentgenium" ),  112 : ( "Cn", "Copernicium" ),  113 : ( "Uut", "Ununtrium" ),  114 : ( "Fl", "Flerovium" ),
    115 : ( "Uup", "Ununpentium" ), 116 : ( "Lv", "Livermorium" ),  117 : ( "Uus", "Ununseptium" ), 118 : ( "Uuo", "Ununoctium" ) }

elementsSymbolZ = {}
for Z in elementsZSymbolName : elementsSymbolZ[elementsZSymbolName[Z][0]] = Z

elementsNameZ = {}
for Z in elementsZSymbolName : elementsNameZ[elementsZSymbolName[Z][1]] = Z

# temporary names that have since been changed
legacyNames = {
    114 : ("Uuq", "Ununquadium"),
    116 : ("Uuh", "Ununhexium"),
}

def elementSymbolFromZ( Z ) :

    Z = int( Z )
    return( elementsZSymbolName[Z][0] )

def elementNameFromZ( Z ) :

    Z = int( Z )
    return( elementsZSymbolName[Z][1] )

def elementZFromSymbol( symbol ) :

    if symbol in elementsSymbolZ:
        return elementsSymbolZ[symbol]
    for Z,(Symbol,Name) in list(legacyNames.items()):
        if Symbol==symbol:
            # FIXME print warning about deprecated name?
            return Z
    raise KeyError("Element with symbol '%s' not found" % symbol)

def elementZFromName( name ) :

    if name in elementsNameZ:
        return elementsNameZ[name]
    for Z,(Symbol,Name) in list(legacyNames.items()):
        if Name==name:
            # FIXME print warning about deprecated name?
            return Z
    raise KeyError("Element with name '%s' not found" % name)

def nucleusNameFromZAndA( Z, A ) :

    A = int( A )
    if( A < 0 ) : raise Exception( 'Invalid A = %s for Z = %s' % ( A, Z ) )
    elif( A == 0 ) : AStr = '_natural'
    else: AStr = str( A )

    Z = int( Z )
    if( Z == 0 ) :
        if( A != 1 ) : raise Exception( 'Invalid A = %s for neutron (Z = 0)' % A )
        AStr = ''

    return( elementSymbolFromZ( Z ) + AStr )

def nucleusNameFromZA( ZA ) :

    if type(ZA) in (int,float):
        return( nucleusNameFromZAndA( ZA // 1000, ZA % 1000 ) )
    elif type(ZA) is str:
        # should be of form 'zaZZZAAA_suffix'
        Z,A = divmod( int(ZA[2:8]),1000 )
        sym = elementSymbolFromZ(Z)
        suffix = ZA[8:].strip()
        if A==0: A = '_natural'
        if suffix:
            if suffix=='m': suffix='m1'
            return '%s%s_%s' % (sym,A,suffix)
        return '%s%s' % (sym,A)

def getZAOrNameAs_xParticle( particle ) :

    from fudge.gnd import xParticle
    particle_ = particle
    if( type( particle_ ) == type( 1 ) ) : particle_ = nucleusNameFromZA( particle_ )
    if( type( particle_ ) != type( "" ) ) : raise Exception( "Invalid particle = %s type" % particle )
    if( particle_ == 'gamma' ) :
        return( xParticle.photon( ) )
    else :
        return( xParticle.isotope( particle_, getMassFromName( particle_ ) ) )

def getMassFromName( name ) :

    from brownies.legacy.endl.structure import masses
    if( name == 'n' ) : return( masses.getMassWithUnitFromZA( 1 ) )
    if( '_natural' in name ) : 
        symbol, A = name.split( '_' )[0], 0
    else :
        for i, l in enumerate( name ) :
            if( l.isdigit( ) ) : break
        symbol, AStr = name[:i], name[i:]
        for i, l in enumerate( AStr ) :
            if( not( l.isdigit( ) ) ) : break
        A = int( AStr[:i+1] )
    Z = elementsSymbolZ[symbol]
    return( masses.getMassWithUnitFromZA( 1000 * Z + A ) )

def getZandAFromName( name ) :

    if( name == 'n' ) : return( ( 0, 1 ) )
    for i, l in enumerate( name ) :
        if( l.isdigit( ) ) : break
    symbol, AStr = name[:i], name[i:]
    if '_' in AStr: AStr=AStr.split('_')[0]
    Z = elementsSymbolZ[symbol]
    for i, l in enumerate( AStr ) :
        if( not( l.isdigit( ) ) ) : break
    A = int( AStr[:i+1] )
    return (Z,A)

def getZAFromName( name ):
    Z,A=getZandAFromName(name)
    return  Z*1000+int(A)

def getZ_A_suffix_andZAFromName( name ) :
    """Returns the tuple (Z, A, suffix, ZA) for an gnd isotope name (e.g., gnd name = 'Am242_m1'
    returns ( 95, 242, 'm1', 95242 )."""

    if( name == 'n' ) : return( 0, 1, '', 1 )
    if( name == 'gamma' ) : return( 0, 0, '', 0 )
    if( name[:18] == 'FissionProductENDL' ) :
        ZA = int( name[18:] )
        Z = ZA // 1000
        A = 1000 * Z - ZA
        return( Z, A, '', ZA )
    if( '__' in name ) : raise Exception ( "Name = %s" % name )
    naturalSuffix = ''
    if( '_' in name ) :         # Isotope names can have level designator (e.g., 'O16_e3') and naturals are of the form 'S_natural' or 'S_natural_l'
        s = name.split( '_' )   # where S is element's symbol and l is level designator (e.g., 'Xe_natural' or 'Xe_natural_c').
        sZA, suffix = s[:2]
        if( len( s ) > 2 ) :
            if( ( len( s ) > 3 ) or ( suffix != 'natural' ) ) : raise Exception( 'Invalid name for endl ZA particle = %s' % name )
            naturalSuffix = s[2]
    else :
        sZA = name
        suffix = ''
    for i, c in enumerate( sZA ) :
        if( c.isdigit( ) ) : break
    if( not c.isdigit( ) ) : i += 1
    sZ, sA = sZA[:i], sZA[i:]
    Z = elementZFromSymbol( sZ )
    if( Z is None ) : raise Exception( 'No element symbol for particle named %s' % name )
    if( sA == '' ) :
        if( suffix == 'natural' ) : return( Z, 0, naturalSuffix, 1000 * Z )
        if( suffix == '' ) : return( Z, 0, '', 1000 * Z )
        raise Exception( 'No A for particle named %s' % name )
    elif( suffix == 'natural' ) :
        raise Exception( 'Natural element also has A defined for particle named %s' % name )
    else :
        try :
            A = int( sA )
        except :
            raise Exception( 'Could not convert A to an integer for particle named %s' % name )
    ZA = 1000 * Z + A
    return( Z, A, suffix, ZA )

# Crack the isotope name to get the A & symbol
def elementAFromName( name ):
    sym = ''
    A = ''
    if '_' in name: m = name.split('_')[1]
    else: m = None
    for c in name.split('_')[0]:
        if c.isalpha(): sym += c
        else: A += c
    if sym == 'n': return sym, 1, None
    if sym == 'g': return sym, 0, None
    if m =='natural': A = '0'
    return sym, A, m
