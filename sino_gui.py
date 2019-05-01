# Copyright (C) 2019 Paul King

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

""" **Graphical User Interface** """

# pylint: disable=E1101,F0401

import os
import copy
import sys

from typing import Callable
from scipy import interpolate

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from functools import partial
import PIL

import sino_funct as sinogram

class GUI(tk.Frame):
    """ Graphical User Interface for Sinogram Class

    """

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        parent.wm_title("Sinogram Tool")
        parent.resizable(False, False)

        selector_frame = tk.Frame(self, width=5, height=100, background="bisque")
        graph_frame = tk.Frame(self, width=90, height=100, background="bisque")

        selector_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        fig = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.subplot = fig.add_subplot(111)
        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.status = tk.StringVar()
        self.status_bar = tk.Frame(master=graph_frame, relief=tk.RIDGE, background="bisque")
        self.status_label=tk.Label(self.status_bar, bd=1, relief=tk.FLAT, anchor=tk.W,
                                   textvariable=self.status, background="bisque",
                                   font=('arial',10,'normal'))
        self.status_label.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.status_bar.pack(fill=tk.X, expand=False, side=tk.LEFT)

        menu = tk.Menu(parent)
        parent.config(menu=menu)
        ## ----------
        _file = tk.Menu(menu)
        __from = tk.Menu(_file)
        __to = tk.Menu(_file)
        _edit = tk.Menu(menu)
        _get = tk.Menu(menu)
        _help = tk.Menu(menu)
        ## ----------
        menu.add_cascade(label="File", menu=_file)
        _file.add_cascade(label='From ...', menu=__from)
        __from.add_command(label="CSV", command=self.from_csv)
        __from.add_command(label="BIN", command=self.from_bin)
        _file.add_cascade(label='To ...', menu=__to)
        __to.add_command(label="PNG", command=self.to_png)
        __to.add_command(label="UnshuffPDF", command=self.to_unshuff_pdf)
        _file.add_command(label="Clear", command=self.file_clr)
        _file.add_command(label="Exit", command=self._quit)
        menu.add_cascade(label="Edit", menu=_edit)
        _edit.add_command(label="Crop", command=self.crop)
        menu.add_cascade(label="Get", menu=_get)
        _get.add_command(label="Mod Factor", command=self.get_mod_fact)
        _get.add_command(label="Histogram", command=self.get_histogram)        
        menu.add_cascade(label="Help", menu=_help)
        _help.add_command(label="About...", command=self.about)

        self.sinogram = np.zeros((200,64), dtype=float)
        self.data_folder = os.path.abspath(
            os.path.join(os.path.split(__file__)[0], 'test_data'))

        self.update('__init__')
        self.canvas.draw()

    def update(self, msg):
        self.subplot.cla()
        try:
            self.subplot.imshow(self.sinogram.data, cmap='gist_yarg')
        except IndexError:
            pass
        self.status.set(msg)
        self.canvas.draw()

    def from_csv(self):
        filename = askopenfilename(
            initialdir=self.data_folder, title="CSV Sinogram",
            filetypes=(("CSV Files", "*.csv"), ("all files", "*.*")))
        self.sinogram = sinogram.from_csv(filename)
        self.update('from_csv')

    def from_bin(self):
        filename = askopenfilename(
            initialdir=self.data_folder, title="BIN Sinogram",
            filetypes=(("BIN Files", "*.bin"), ("all files", "*.*")))
        self.sinogram = sinogram.from_bin(filename)
        self.update('from_bin')

    def to_png(self):
        filename = asksaveasfilename(
            initialdir=self.data_folder, title="PNG Image",
            filetypes=(("PNG Files", "*.png"), ("all files", "*.*")))
        self.sinogram = sinogram.to_png(self.sinogram, filename)
        self.update('to_png')

    def to_unshuff_pdf(self):
        try:
            initialfile = self.sinogram.meta['document_id']
        except KeyError:
            initialfile = ''
        filename = asksaveasfilename(
            initialdir=self.data_folder, title="PDF Document",
            initialfile=initialfile,
            filetypes=(("PDF Files", "*.pdf"), ("all files", "*.*")))
        self.sinogram = sinogram.to_unshuff_pdf(self.sinogram, filename)
        self.update('to_unshuff_pdf')

    def file_clr(self):
        self.sinogram = np.zeros((200,64), dtype=float)
        self.update('file_clr')

    def crop(self):
        self.sinogram = sinogram.crop(self.sinogram)
        self.update('crop')

    def get_mod_fact(self):
        mod_factor = sinogram.get_mod_factor(self.sinogram)
        result = "Modulation Factor: {0:.3f}".format(mod_factor)
        self.update(result)

    def get_histogram(self):
        try:
            initialfile = self.sinogram.meta['document_id']
        except KeyError:
            initialfile = ''
        filename = asksaveasfilename(
            initialdir=self.data_folder, title="PNG Image",
            initialfile=initialfile,
            filetypes=(("PNG Images", "*.png"), ("all files", "*.*")))
        sinogram.get_histogram(self.sinogram, bins=10, file_name=initialfile)
        self.update('get_histogram')

    def on_key_press(self, event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)

    def _quit(self):
        root.quit()
        root.destroy()

    def about(self):
        tk.messagebox.showinfo(
            "About", "Sinogram Tool \n king.r.paul@gmail.com")

if __name__ == "__main__":
    root = tk.Tk()
    GUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()