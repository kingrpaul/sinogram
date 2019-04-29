# Copyright (C) 2018 Paul King

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version (the "AGPL-3.0+").

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License and the additional terms for more
# details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# ADDITIONAL TERMS are also included as allowed by Section 7 of the GNU
# Affero General Public License. These additional terms are Sections 1, 5,
# 6, 7, 8, and 9 from the Apache License, Version 2.0 (the "Apache-2.0")
# where all references to the definition "License" are instead defined to
# mean the AGPL-3.0+.

# You should have received a copy of the Apache-2.0 along with this
# program. If not, see <http://www.apache.org/licenses/LICENSE-2.0>.


""" For importing, analyzing, and converting tomotherapy sinograms."""

# pylint: disable=E1101

from string import digits as DIGITS
from string import ascii_letters as LETTERS
import csv
import os
import re, sys
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

class Sinogram():
    """ array of 64-element projections arrays
    
   Leaf-open-times (0->1) collected into 64-element projection arrays,
   which are then collected into the Sinogram array of arbitrary length.

    Attributes
    ----------
    data : np.array
    meta : dict, optional
    shape: 2-tuple

    """

    def __init__(self, data, meta={}):
        """ instantiation """
        if type(data) != np.ndarray:
            data = np.array(data)
        self.data = data
        self.meta = meta
        self.shape = self.data.shape

        try: 
            assert 0 <= self.shape[1] <= 64
            self.meta['cropped'] = self.shape[1] < 64
        except AssertionError:
            raise ValueError('invalid number of mlc leaves')

    def __str__(self):
        """
        Examples
        --------
        ``Sinogram object: 464 projs | 64 leaves | open time (0.0 -> 1.0)``
        """
        try:
            fmt_str = 'Sinogram object: {} projs | {} leaves | open time ({} -> {})'
            return fmt_str.format(len(self.data),
                                  len(self.data[0]),
                                  np.min(self.data), np.max(self.data))
        except ValueError:
            return ''  # EMPTY SINOGRAM


def from_csv(file_name):
    """ get sinogram from csv file

    Produced by reading a RayStation sinogram CSV file.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    sinogram : np.ndarray-like

    Notes
    -----
    As produced by ExportTomoSinogram.py, Brandon Merz, RaySearch 
    customer forum, 1/18/2018. File first row contains demographics. 
    Subsequent rows correspond to couch positions. 

    """

    with open(file_name, 'r') as csvfile:
        pat_id = csvfile.readline()
        meta = {'document_id': re.search(r'ID: (\d*)', pat_id).group(1)  +\
                       ' - ' + re.search(r'name: (\w*)\^', pat_id).group(1) +\
                        ', ' + re.search(r'\^(.*), ID:', pat_id).group(1)}
        reader = csv.reader(csvfile, delimiter=',')
        data = np.asarray([line[1:] for line in reader]).astype(float)
    return Sinogram(data, meta=meta)


def from_bin(file_name):
    """ read sinogram from binary file

    Produced by Accuray sinogram BIN files, as used in calibration plans.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    sinogram : np.array-like

    """

    data = np.fromfile(file_name, dtype=float, count=-1, sep='')
    data = np.reshape(data, (len(data)//64, 64))
    return Sinogram(data)

def to_png(sinogram, file_name):
    """ get png image file from sinogram
    
    Parameters
    ----------
    sinogram : np.ndarray-like
    file_name : str

    Returns
    -------
    None

    """
    fig = plt.figure(figsize=(len( sinogram.data[0])/100, 
                              len( sinogram.data)/100), 
                              dpi=1000)
    grid_spec = GridSpec(nrows=1, ncols=1)
    subplot = fig.add_subplot(grid_spec[0])
    subplot.imshow( sinogram.data, cmap='gist_yarg')
    subplot.axes.get_xaxis().set_visible(False)
    subplot.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.splitext(file_name)[0] + '.png')
    plt.close()
    

def crop(sinogram):
    """ crop sinogram

    Return a symmetrically cropped sinogram, such that always-closed
    leaves are excluded and the sinogram center is maintained.

    Parameters
    ----------
    sinogram : np.array

    Returns
    -------
    sinogram : np.array

    """

    idx =  list(np.any(sinogram.data > 0, axis=0))
    idx =  idx or idx[::-1]  # SYMMETRIZE
    data = sinogram.data[:, idx]
    meta = {'cropped': True}
    return Sinogram(data, meta=meta)


def unshuffle(sinogram):
    """ unshuffle singram by angle

    Return a list of 51 sinograms, by unshuffling the provided
    sinogram; so that all projections in the result correspond
    to the same gantry rotation angle, analogous to a fluence map.

    Parameters
    ----------
    sinogram : np.array

    Returns
    -------
    unshuffled: list of sinograms

    """
    unshufd = [[] for i in range(51)]
    idx = 0
    for prj in sinogram:
        unshufd[idx].append(prj)
        idx = (idx + 1) % 51
    return unshufd


def make_histogram(sinogram, bins=10, file_name=''):
    """ make a leaf-open-time histogram

    Return a histogram of leaf-open-times for the provided sinogram. If a 
    filename is provided, then a grapn of the histogram is saved at that
    location. 

    Parameters
    ----------
    sinogram : np.array
    bins : int, optional
    file_name : string, optional

    Returns
    -------
    histogram : np.array

    """

    if file_name:
        plt.hist(sinogram.data[sinogram.data>0].flatten(), bins=bins)
        plt.savefig(os.path.splitext(file_name)[0] + '.png')
        plt.close()

    rng = (0,np.max(sinogram.data))
    return np.histogram(sinogram.data, bins=bins, range=rng)


def get_mod_factor(sinogram):
    """ modulation factor 

    Ratio of max to mean leaf open time, over all non-zero values.

    Parameters
    ----------
    sinogram : np.array

    Returns
    -------
    modulation factor : float

    """
    return np.max(sinogram.data) / np.mean(sinogram.data[sinogram.data>0])

if __name__ == "__main__":
    import gui
    import tkinter as tk
    root = tk.Tk()
    gui.GUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()