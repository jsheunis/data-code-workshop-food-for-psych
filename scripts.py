import nibabel as nib
import numpy as np
from niwidgets import NiftiWidget
from nipy.algorithms.registration import HistogramRegistration, Rigid, resample
import warnings; warnings.simplefilter('ignore')


def estimateMotion(niiRefVol, niiVol, mthd):
    
    reg = HistogramRegistration(niiVol, niiRefVol, interp='tri')

    # estimate optimal transformation
    T = reg.optimize('rigid', optimizer=mthd)

    # get the realignment parameters:
    rot_x, rot_y, rot_z = np.rad2deg(T.rotation)
    trans_x, trans_y, trans_z = T.translation

    motionParams = np.array([trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]) # Stephan: switched order wrt original code

    realignedVolume = resample(niiVol, T.inv(), niiRefVol)

    return motionParams, realignedVolume