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
    import pymedphys_labs.paulking.sinogram as sinogram

elif 'sinogram' in __file__:
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    import sinogram

DATA_PATH = os.path.join(os.getcwd(), 'tests', 'data')
assert os.path.isdir(DATA_PATH)

SIN_CSV_FILE = os.path.join(DATA_PATH, "sinogram.csv")
assert os.path.isfile(SIN_CSV_FILE)

SIN_BIN_FILE = os.path.join(DATA_PATH, 'MLC_all_test_old_800P.bin')
assert os.path.isfile(SIN_BIN_FILE)

PNG_CSV_FILE = os.path.join(DATA_PATH, 'csv_to_png_result.png')
PNG_BIN_FILE = os.path.join(DATA_PATH, 'bin_to_png_result.png')
for f in [PNG_CSV_FILE, PNG_BIN_FILE]:
    if os.path.isfile(f):
        os.remove(f)

def test_from_csv():
    result = sinogram.from_csv(SIN_CSV_FILE)
    assert result.meta['document_id'] == '00000 - ANONYMOUS, PATIENT'
    assert result.shape == (464, 64)
    assert np.all(result.data <= 1.0)
    assert np.all(result.data >= 0.0)

def test_from_bin():
    result = sinogram.from_bin(SIN_BIN_FILE)
    assert result.shape == (400, 64)
    assert np.all(result.data <= 1.0)
    assert np.all(result.data >= 0.0)

def test_csv_to_png():
    sinogram.to_png(sinogram.from_csv(SIN_CSV_FILE), PNG_CSV_FILE)
    assert os.path.isfile(PNG_CSV_FILE)

def test_bin_to_png():
    sinogram.to_png(sinogram.from_bin(SIN_BIN_FILE), PNG_BIN_FILE)
    assert os.path.isfile(PNG_BIN_FILE)

def test_crop():
    UNCROPPED = sinogram.Sinogram([[0.0]*31 + [1.0]*2 + [0.0]*31, 
                                   [0.0]*31 + [1.0]*2 + [0.0]*31])
    CORRECT = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.all(sinogram.crop(UNCROPPED).data == CORRECT)

def test_unshuffle():
    unshuffled = sinogram.unshuffle([[0]*25 + [1.0]*14 + [0]*25]*510)
    assert len(unshuffled) == 51          # number of angles
    assert len(unshuffled[0]) == 10       # number of projections
    assert unshuffled[0][0][0] == 0       # first leaf is closed


def test_make_histogram():
    result = sinogram.from_csv(SIN_CSV_FILE).data
    assert np.allclose(sinogram.make_histogram(result)[0][0], [0., 0.1])
    assert sinogram.make_histogram(result)[0][1] == 25894


def test_find_modulation_factor():
    result = sinogram.from_csv(SIN_CSV_FILE).data
    assert np.isclose(sinogram.find_modulation_factor(result), 2.762391)


if __name__ == "__main__":
    test_from_csv()
    test_from_bin()
    test_csv_to_png()
    test_bin_to_png()
    test_crop()
    test_unshuffle()
    test_make_histogram()
    test_find_modulation_factor()
