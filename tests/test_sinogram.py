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

import sys
import os
import numpy as np

if 'pymedphys' in __file__:
    from pymedphys_labs.paulking.sinogram import read_csv_file
    from pymedphys_labs.paulking.sinogram import read_bin_file
    from pymedphys_labs.paulking.sinogram import crop
    from pymedphys_labs.paulking.sinogram import make_histogram
    from pymedphys_labs.paulking.sinogram import find_modulation_factor
    from pymedphys_labs.paulking.sinogram import unshuffle

elif 'sinogram' in __file__:
    sys.path.insert(0, os.path.abspath('.\\src'))
    import sinogram ######
    from sinogram import read_csv_file
    from sinogram import read_bin_file
    from sinogram import crop
    from sinogram import make_histogram
    from sinogram import find_modulation_factor
    from sinogram import unshuffle

SIN_CSV_FILE = os.path.join(
    os.path.dirname(__file__), "./data/sinogram.csv")

SIN_BIN_FILE = os.path.join(
    os.path.dirname(__file__), "./data/MLC_all_test_old_800P.bin")


def test_read_csv_file():
    result = sinogram.read_csv_file(SIN_CSV_FILE)
    assert result.meta['document_id'] == '00000 - ANONYMOUS, PATIENT'
    assert result.shape == (464, 64)
    assert np.all(result.data <= 1.0)
    assert np.all(result.data >= 0.0)

def test_read_bin_file():
    result = read_bin_file(SIN_BIN_FILE)
    assert result.shape == (400, 64)
    assert np.all(result.data <= 1.0)
    assert np.all(result.data >= 0.0)




def test_crop():
    STRIP = [[0.0]*31 + [1.0]*2 + [0.0]*31,
             [0.0]*31 + [1.0]*2 + [0.0]*31]
    assert crop(STRIP) == [[1.0, 1.0], [1.0, 1.0]]


def test_unshuffle():
    unshuffled = unshuffle([[0]*25 + [1.0]*14 + [0]*25]*510)
    assert len(unshuffled) == 51          # number of angles
    assert len(unshuffled[0]) == 10       # number of projections
    assert unshuffled[0][0][0] == 0       # first leaf is closed


def test_make_histogram():
    sinogram = read_csv_file(SIN_CSV_FILE).data
    # sinogram = read_csv_file(SIN_CSV_FILE)[-1]
    assert np.allclose(make_histogram(sinogram)[0][0], [0., 0.1])
    assert make_histogram(sinogram)[0][1] == 25894
    # [(array([0. , 0.1]), 25894),
    #  (array([0.1, 0.2]), 0),
    #  (array([0.2, 0.3]), 11),
    #  (array([0.3, 0.4]), 3523), ...]


def test_find_modulation_factor():
    sinogram = read_csv_file(SIN_CSV_FILE).data
    # sinogram = read_csv_file(SIN_CSV_FILE)[-1]
    assert np.isclose(find_modulation_factor(sinogram), 2.762391)


if __name__ == "__main__":
    test_read_csv_file()
    test_read_bin_file()
    test_crop()
    test_unshuffle()
    test_make_histogram()
    test_find_modulation_factor()
