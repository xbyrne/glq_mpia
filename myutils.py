"""
Some utilities for the project
"""
from astropy import units as u

## Converting Vega to AB mags

# See https://www.eso.org/rm/api/v1/public/releaseDescriptions/144
#  and https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
delta_m = {
    'J': 0.916,
    'K': 1.827,
    'W1': 2.699,
    'W2': 3.339
}
def vega_to_AB(vega_mag, band):
    """
    Converts Vega magnitudes to AB system
    """
    return vega_mag + delta_m[band]

def AB_to_uJy(mag_AB):
    """
    Converts an AB magnitude to a flux in uJy.
    """
    return 10**(29-(48.60/2.5)) * 10**(-mag_AB/2.5)