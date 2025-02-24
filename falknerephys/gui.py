import sys
from importlib import resources
import h5py
import numpy as np
from nptdms import TdmsFile
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from matplotlib.collections import PathCollection
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from falknerephys.io.utils import find_files
from falknerephys.classes import MDcontroller


def get_data_dialog(filter=None):
    """
    Opens a file dialog to select a directory.

    Returns
    -------
    str
        Path to the selected directory.
    """
    dialog = QFileDialog()
    if filter is None:
        data_fold = dialog.getExistingDirectory()
    else:
        data_fold = dialog.getOpenFileName(filter=filter)
    return data_fold

def get_save_dialog(filter='', suffix=''):
    """
    Opens a file dialog to select a save file path.

    Parameters
    ----------
    filter : str, optional
        Filter for the file dialog (default is '').
    suffix : str, optional
        Suffix to add to the file name (default is '').

    Returns
    -------
    str or None
        Path to the selected save file or None if no file was selected.
    """
    dialog = QFileDialog()
    save_path = dialog.getSaveFileName(filter=filter)
    if save_path[1] != '':
        out_path = save_path[0] + suffix + save_path[1]
        return out_path
    else:
        return None

class EphysApp:
    def __init__(self, data_folder: str = None):
        """
        Initializes the PtectApp. Starts the QApplication process

        Parameters
        ----------
        data_folder : str, optional
            Path to the data folder (default is None).
        """
        matplotlib.use('QT5Agg')
        matplotlib.rcParams.update({'font.size': 12})
        self.app = QApplication(sys.argv)
        self.app.setStyleSheet("QObject{font-size: 12pt;}")
        self.gui = EphysWindowManager(data_folder=data_folder)
        sys.exit(self.app.exec())


class EphysController:
    def __init__(self, data_folder: str = None):
        """
        Initializes the EphysController.

        Parameters
        ----------
        data_folder : str, optional
            Path to the data folder (default is None).
        """
        if data_folder is None:
            data_folder = get_data_dialog()
        self.data_folder = data_folder
        self.ephys_files = find_files(data_folder)
        self.metadata = MDcontroller(self.ephys_files['metadata.yml'])

    def get_info(self):
        """
        Retrieves the metadata information as a string.

        Returns
        -------
        str
            Metadata information.
        """
        return str(self.metadata)

    def save_info(self):
        """
        Saves the metadata information.
        """
        self.metadata.save_metadata(self.metadata.file_name)

    def get_file_list(self):
        """
        Retrieves the list of territory files.

        Returns
        -------
        dict
            Dictionary containing the territory files.
        """
        return self.ephys_files


class EphysWindow(QWidget):

    def __init__(self, parent=None, ephys_cont: EphysController=None, qt_signal=None):
        """
        Initializes a generic PtectWindow with logo.

        Parameters
        ----------
        ptect_cont : PtectController, optional
            Controller for the Peetector (default is None).
        parent : QWidget, optional
            Parent widget (default is None).
        """
        super().__init__()
        self.parent = parent
        self.control = ephys_cont
        self.signals = qt_signal


class EphysWindowManager(QMainWindow):
    signals = pyqtSignal(str)
    def __init__(self, parent=None, data_folder=None):
        """
        Initializes the PtectMainWindow, which manages each subwindow

        Parameters
        ----------
        data_folder : str, optional
            Path to the data folder (default is None).
        """
        super().__init__()
        self.parent = parent
        self.signals.connect(self.read_signals)
        self.windows = []
        self.spawn_window(EphysMenuWindow(self, qt_signal=self.signals))

    def spawn_window(self, window: EphysWindow=None):
        if window is None:
            window = EphysWindow(self)
        self.windows.append(window)
        window.show()
        return window

    def close_all(self):
        for window in self.windows:
            window.hide()
            window.deleteLater()
        self.windows = []

    def read_signals(self, message):
        parsing = message.split(':')
        match parsing[0]:
            case 'open':
                self.switch_window(parsing[1])
            case 'close':
                self.switch_window('menu')
        print(message)

    def switch_window(self, window_type: str):
        new_win = None
        match window_type:
            case 'daq':
                new_win = EphysDaqWindow(qt_signal=self.signals)
            case 'menu':
                new_win = EphysMenuWindow(qt_signal=self.signals)
        if new_win is not None:
            self.close_all()
            self.spawn_window(new_win)


class EphysDaqWindow(EphysWindow):
    def __init__(self, parent=None, qt_signal=None, daq_file=None):
        super().__init__(parent, qt_signal=qt_signal)
        if daq_file is None:
            daq_file, _ = get_data_dialog(filter='*.tdms')
        self.daq_file = daq_file
        self.load_data()

    def closeEvent(self, event):
        """
        Handles the close event for the run window.

        Parameters
        ----------
        event : QCloseEvent
            Close event.
        """
        self.signals.emit('close:daq')

    def load_data(self):
        file_type = self.daq_file.split('.')[-1]
        match file_type:
            case 'tdms':
                tdms_file = TdmsFile(self.daq_file)
                print(tdms_file.read_metadata())


class EphysMenuWindow(EphysWindow):
    def __init__(self, parent=None, qt_signal=None):
        super().__init__(parent, qt_signal=qt_signal)
        self.layout = QVBoxLayout()
        self.init_controls()
        self.setLayout(self.layout)

    def init_controls(self):
        but_names = ['Load Daq', 'Load Ephys', 'Demo']
        but_funcs = [self.load_daq_cb, self.temp_cb, self.temp_cb]
        for name, func in zip(but_names, but_funcs):
            but = QPushButton(name)
            but.clicked.connect(func)
            self.layout.addWidget(but)

    def temp_cb(self):
        self.signals.emit('temp_cb')

    def load_daq_cb(self):
        self.signals.emit('open:daq')


class EphysDataWindow(EphysWindow):
    def __init__(self, parent=None, control=None):
        """
        Initializes the PtectDataWindow.

        Parameters
        ----------
        ptect_cont : PtectController
            Controller for the Peetector.
        parent : QWidget, optional
            Parent widget (default is None).
        """
        super().__init__(parent, control)
        self.resize(640, 480)
        self.setWindowTitle('Output from: ' + self.control.data_folder)
        self.grid = QGridLayout(self)
        self.setLayout(self.grid)


class SlideInputer(QGroupBox):
    def __init__(self, name, label=None, low=0, high=255):
        """
        Initializes the SlideInputer.

        Parameters
        ----------
        name : str
            Name of the slider.
        label : str, optional
            Label for the slider (default is None).
        """
        if label is None:
            super().__init__(name)
        else:
            super().__init__(label)
        self.id = name
        slide_group = QVBoxLayout()
        self.slide = QSlider()
        self.slide.setMinimum(low)
        self.slide.setMaximum(high)
        self.slide.valueChanged.connect(self.update_ebox)

        self.ebox = QLineEdit()
        self.ebox.setValidator(QIntValidator())
        self.ebox.setMaxLength(3)
        self.ebox.textChanged.connect(self.update_slide)

        for w in (self.slide, self.ebox):
            slide_group.addWidget(w)
        self.setLayout(slide_group)

    def update_slide(self, val):
        """
        Updates the slider value.

        Parameters
        ----------
        val : str
            Value to set for the slider.
        """
        if len(val) > 0:
            val = int(val)
            self.slide.setValue(val)

    def update_ebox(self, val):
        """
        Updates the edit box value.

        Parameters
        ----------
        val : int
            Value to set for the edit box.
        """
        val = str(val)
        self.ebox.setText(val)

    def get_value(self):
        """
        Retrieves the current value of the slider.

        Returns
        -------
        tuple
            Tuple containing the slider ID and value.
        """
        return self.id, self.slide.value()

    def set_value(self, value):
        self.slide.setValue(value)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        """
        Initializes an MplCanvas to display Matplotlib plots via QTAgg
        """
        self.fig = plt.Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        """
        Initializes the PlotWidget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (default is None).
        """
        QWidget.__init__(self, parent)
        self.current_pobj = []
        self.canvas = MplCanvas()
        pyqt_grey = (240/255, 240/255, 240/255)
        self.canvas.fig.set_facecolor(pyqt_grey)
        self.canvas.ax.set_facecolor(pyqt_grey)
        self.canvas.ax.spines['top'].set_visible(False)
        self.canvas.ax.spines['right'].set_visible(False)
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = matplotlib.cm.ScalarMappable(norm=norm)
        self.color_bar = self.canvas.fig.colorbar(sm, ax=self.gca())
        self.color_bar.ax.set_visible(False)

    def gca(self):
        """
        Retrieves the current axis from the MplCanvas

        Returns
        -------
        matplotlib.axes.Axes
            Current axis.
        """
        return self.canvas.ax

    def clear(self):
        """
        Clears the plot.
        """
        if len(self.current_pobj) > 0:
            for pobj_list in self.current_pobj:
                if type(pobj_list) is PathCollection:
                    pobj_list.remove()
                else:
                    for pobj in pobj_list:
                        pobj.remove()
            self.current_pobj = []

    def plot(self, *args, plot_style=None, **kwargs):
        """
        Plots the data.

        Parameters
        ----------
        *args : tuple
            Data to plot.
        plot_style : str, optional
            Style of the plot (default is None).
        **kwargs : dict
            Additional keyword arguments for the plot. Passed to matplotlib.plot
        """
        my_ax = self.gca()
        if plot_style is not None:
            if plot_style == 'scatter':
                self.current_pobj.append(my_ax.scatter(args[0], args[1], *args[2:], **kwargs))
        else:
            self.current_pobj.append(my_ax.plot(*args, **kwargs))
        # self.canvas.draw()

    def draw(self):
        """
        Draws the plot.
        """
        self.canvas.draw()

    def colorbar(self, min_val, max_val, label='', orient='vertical', cmap='summer'):
        """
        Adds a colorbar to the plot.

        Parameters
        ----------
        min_val : float
            Minimum value for the colorbar.
        max_val : float
            Maximum value for the colorbar.
        label : str, optional
            Label for the colorbar (default is '').
        orient : str, optional
            Orientation of the colorbar (default is 'vertical').
        cmap : str, optional
            Colormap for the colorbar (default is 'summer').
        """
        # for a in self.canvas.fig.axes:
        #     if a.label == 'colorbar':
        #         self.canvas.fig.axes.remove(a)
        # self.color_bar.remove()

        self.color_bar.ax.set_visible(True)
        norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        self.color_bar.update_normal(sm)
        self.color_bar.set_label(label)

        # self.color_bar = self.canvas.fig.colorbar(sm,
        #              ax=self.gca(), orientation=orient, label=label)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EphysWindowManager()
    sys.exit(app.exec())
