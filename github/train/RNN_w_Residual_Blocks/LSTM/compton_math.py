from numpy import power, sqrt, cos, dot
from numpy import power as p
from numpy import arccos as acos
from numpy import zeros, arange, isnan, array
from numpy import pi, where, nan
import numpy

from numpy import matmul
from numpy.linalg import norm

# abnormal e0 could be due to double to triple events which are not
# beholden to the nice formulas
# The same goes for false events

# thresholds
# expect gamma rays in [1/2 MeV , 7ish MeV]
# Due a histogram of energy to see more common spikes.

# Either way, it would be safer to stick with cosine of the angle
# If for sure we see that the arccos problem is expected and not a
# mathematical bug.

def getE0(df):
    """(dataframe of scatter data) -> dataframe
    
    Takes in a dataframe of scatters
    """
    moc2 = 0.511 # MeV

    d = p(df['e2'], 2) + (4*df['e2']*moc2)/(1-cos(df['scatter2']))

    e0 = df['e1'] + 0.5*(df['e2']+sqrt(d))
    df['e0'] = e0
    return df

def getScatter2(df):
    # Get triangle angle between scatter 1 and scatter 2
    # Uses the law of cosines to compute the angle.
    # Shift the far point to 0
    vectorTip1 = df[['x3', 'y3', 'z3']].to_numpy() - df[['x2', 'y2', 'z2']].to_numpy()
    vectorTip2 = df[['x2', 'y2', 'z2']].to_numpy() - df[['x1', 'y1', 'z1']].to_numpy()
    len1    = norm(vectorTip1)
    len2    = norm(vectorTip2)

    i = arange(vectorTip1.shape[0])
    # Regular matrix multiplication works
    d = (vectorTip1 * vectorTip2).sum(axis=1)
    # angle = matmul(vectorTip1,vectorTip2.T)[i,i]/(len1*len2)
    angle = d/(len1*len2)
    angle = acos(angle)
    # print(angle.shape)
    # innerAngle  = p(df['euc1'], 2) + p(df['euc2'], 2) - p(df['euc3'], 2)
    # innerAngle  = innerAngle / (2*df['euc1'].multiply(df['euc2']))
    # innerAngle  = acos(innerAngle)
    # The scattering angle is supplementary angle between the first and second
    # scatter
    # scat2 = pi-innerAngle
    df['scatter2'] = angle
    return df

def getScatter1(df):
    moc2 = 0.511 # MeV
    scat1 = 1 + moc2*(
                p(df['e0'], -1) - p((df['e0']-df['e1']), -1)
            )
    print("b", scat1.shape, scat1.min(), scat1.max())
    print("b e0", df['e0'].min(), df['e0'].max())
    # scat1 = acos(scat1)
    # scat1 = acos(scat1)
    df['scatter1'] = scat1
    return df

def getETotal(df):
    df['ET'] = df['e1'] + df['e2'] + df['e3']
    return df


def euc(df_scatters):
    """
    
    Computes the euclidean distances between scatters by event.
    """
    new_df = df_scatters.copy()
    for euc in [1, 2, 3]:
        euc = "euc{}".format(euc)
        if euc not in new_df.columns:
            new_df[euc] = numpy.NaN
    # Do this for all events
    # Doubles -- We know that e3 is null for doubles...
    new_df.loc[new_df['e3'].isnull(), ['euc1']] = \
        sqrt(power( new_df.loc[new_df['e3'].isnull(), ['x1', 'y1', 'z1']].to_numpy() 
                  - new_df.loc[new_df['e3'].isnull(), ['x2', 'y2', 'z2']].to_numpy(), 2).sum(axis=1))
    new_df.loc[new_df['e3'].isnull(), ['euc2']] = \
        sqrt(power( new_df.loc[new_df['e3'].isnull(), ['x1', 'y1', 'z1']].to_numpy() 
                  - new_df.loc[new_df['e3'].isnull(), ['x2', 'y2', 'z2']].to_numpy(), 2).sum(axis=1))
    # Triples -- We know that e3 is not null for triples
    new_df.loc[new_df['e3'].notnull(), ['euc1']] = \
        sqrt(power(new_df.loc[new_df['e3'].notnull(), ['x1', 'y1', 'z1']].to_numpy()
                  -new_df.loc[new_df['e3'].notnull(), ['x2', 'y2', 'z2']].to_numpy(), 2).sum(axis=1))
    new_df.loc[new_df['e3'].notnull(), ['euc2']] = \
        sqrt(power(new_df.loc[new_df['e3'].notnull(), ['x2', 'y2', 'z2']].to_numpy()
                  -new_df.loc[new_df['e3'].notnull(), ['x3', 'y3', 'z3']].to_numpy(), 2).sum(axis=1))
    new_df.loc[new_df['e3'].notnull(), ['euc3']] = \
        sqrt(power(new_df.loc[new_df['e3'].notnull(), ['x3', 'y3', 'z3']].to_numpy()
                  -new_df.loc[new_df['e3'].notnull(), ['x1', 'y1', 'z1']].to_numpy(), 2).sum(axis=1))
    return new_df

def diffe(df_scatters):
    df_scatters['de1'] = df_scatters['e2'].subtract(df_scatters['e1'])#.abs()
    df_scatters['de2'] = df_scatters['e3'].subtract(df_scatters['e2'])#.abs()
    df_scatters['de3'] = df_scatters['e1'].subtract(df_scatters['e3'])#.abs()
    return df_scatters

def shiftData(dat):
    # Shift all data to the origin
    xmin = dat[['x1', 'x2','x3']].min().min()
    ymin = dat[['y1', 'y2','y3']].min().min()
    zmin = dat[['z1', 'z2','z3']].min().min()
    dat[['x1', 'x2', 'x3']] -= xmin
    dat[['y1', 'y2', 'y3']] -= ymin
    dat[['z1', 'z2', 'z3']] -= zmin
    return dat

def shrinkData(dat):
    xmax = dat[['x1', 'x2','x3']].max().max()
    ymax = dat[['y1', 'y2','y3']].max().max()
    zmax = dat[['z1', 'z2','z3']].max().max()
    # Shrink the slice into the unit cube
    shrink = 1.0/max([xmax, ymax, zmax])
    xshrink, yshrink, zshrink = shrink, shrink, shrink
    # xshrink = 1.0/xmax
    # yshrink = 1.0/ymax
    # zshrink = 1.0/zmax
    dat[[ 'x1', 'x2','x3' ]] *= xshrink
    dat[[ 'y1', 'y2','y3' ]] *= yshrink
    dat[[ 'z1', 'z2','z3' ]] *= zshrink
    return dat


# Copied from preprocessing...
