import numpy as np
import matplotlib.pyplot as plt


def get_mids(bins):
    """Calculate the bin mids from an array of bins"""
    return (bins[1:] + bins[:-1]) / 2


def rebin(histo, bins):
    """TODO: update histogram to different binning via RegularGridInterpolator"""
    pass


class Mephistogram:
    """My elegantly programmed histogram class.
    Currently tested for 1D and 2D histograms.


    Attributes:
    -----------
    histo:

    bins:

    bin_mids:

    axis_names:

    ndim:

    shape:


    Methods:
    --------

    match:

    match_matmul:

    T:

    plot:


    """

    def __init__(self, histo_or_nums, bins, axis_names=None, make_hist=False):
        """Initialize the mephistogram with its bin content and bin edges.

        Histogram and bin dimensionality is checked and stored,
        and axis names are assigned or generated.

        Parameters:
        -----------
        histo_or_nums: ndarray
            numpy array of histogram bin contents (make_hist=False)
            OR numpy array of raw numbers from which to calculate the histogram (make_hist=True)
            1D: length n = np.shape(histo)[0]
            2D: shape n x m = np.shape(histo)
        bins: 1D array or tuple of 1D arrays
            bin edges matching the dimensionality of histo:
            1D array with length n+1
            2D tuple of arrays with lengths (n+1, m+1)

        Optional Parameters:
        --------------------
        axis_names: str or tuple of str
            Names of the histogram axis/axes.
            Defaults are 'axis-i' for i in number of axes, i.e. 'axis-0' for 1D histogram.
        make_hist: bool, default False
            Calculate histogram (True) OR directly expect histogram as input (False)

        """
        if make_hist:
            histo_out = self.make_hist(histo_or_nums, bins)
            self.histo = histo_out[0]
        else:
            self.histo = histo_or_nums
        self.ndim = np.ndim(self.histo)
        self.shape = np.shape(self.histo)
        self.set_bins(bins)
        self.set_names(axis_names)

    def unity(self):
        """Neutral element of matmul"""
        pass

    def zeros(self):
        """Neutral element of addition"""
        pass

    def make_hist(self, nums, bins):
        """
        Wrapper for np.histogram or np.histogram2d

        Returns:
        1D case: hist, edges
        2D case: hist, xedges, yedges
        """
        # check input
        if isinstance(bins, tuple):
            assert len(bins) == 2, "bins must be 1D or 2D-tuple"
            assert len(nums) == 2, "num input must be 1D or 2D-tuple"
            return np.histogram2d(*nums, bins=bins)
        else:
            assert bins.ndim == 1, "bins must be 1D or 2D-tuple"
            assert nums.ndim == 1, "num input must be 1D or 2D-tuple"
            return np.histogram(nums, bins=bins)

    def __repr__(self) -> str:
        rep_str = f"Mephistogram with {self.ndim} dimensions and shape {self.shape}."
        rep_str += f"\nAxis names are {self.axis_names}."
        return rep_str

    def set_bins(self, bins):
        """Check that the bin dimensionality matches the histogram and assign to attribute."""
        if self.ndim == 1:  # 1D
            # check the correct bin length
            assert (
                len(bins) == self.shape[0] + 1
            ), "The bin length should match the corresponding histogram axis length."

            self.bin_mids = get_mids(bins)  # calc the bin mids

        else:  # 2D or larger
            # check that we got the correct number of bin edges
            assert self.ndim == len(
                bins
            ), "The number of bins should match the histogram dimensions."

            bin_mids = []
            for idx, b in enumerate(bins):
                # check the correct bin length
                assert len(b) == self.shape[idx] + 1, (
                    f"The bin length ({len(b)-1}) should match "
                    + f"the corresponding histogram axis length ({self.shape[idx]})."
                )

                bin_mids.append(get_mids(b))  # calc the bin mids
            self.bin_mids = tuple(bin_mids)
        # set bins
        self.bins = bins

    def set_names(self, axis_names):
        """Assign axis names, generate them if axis_names is None."""
        # generate default axis names
        if axis_names is None:
            axis_names = (
                "axis-0"
                if self.ndim == 1
                else tuple([f"axis-{i}" for i in range(self.ndim)])
            )
        # check for the correct number of axis names
        if self.ndim == 1:
            assert type(axis_names) is str
        else:
            assert self.ndim == len(
                axis_names
            ), "The number of axis_names should match the histogram dimensions."

        # set names
        self.axis_names = axis_names

    def normalize(self, axis=0):
        if axis == 0:
            self.histo /= np.sum(self.histo, axis=axis)
        elif axis == 1:
            self.histo /= np.sum(self.histo, axis=axis)[:, np.newaxis]
        else:
            raise ValueError("Too many axis dimensions")

    def match_matmul(self, mephisto, verbose=False, raise_err=True) -> bool:
        """Check if two mephistograms are compatible for matrix multiplication.

        Parameters:
        -----------
        mephisto: Mephistogram
            Check if self.histo and Mephistogram.histo are matmul compatible
        verbose: bool, default False
            Do or do not print output
        raise_err: bool, default True
            Raise ValueError if mephistograms are not compatible

        Returns:
        --------
        Bool: compatible or not

        """
        # only for ndim=2
        if self.ndim != 2:
            raise NotImplementedError(
                f"Dimensions are {self.ndim} and {mephisto.ndim}, but should both be 2."
            )
        # last axis of self and first axis of mephisto need to match
        # and the corresponding bins need to be identical
        str_good = "Matrix multiplication possible."
        str_bad = "Matrix multiplication not possible."

        # check matching dimensions for matmul
        if self.shape[-1] != mephisto.shape[0]:
            if raise_err:
                raise ValueError(
                    str_bad + f" Shapes are {self.shape} and {mephisto.shape};"
                )
            elif verbose:
                print(str_bad + f" Shapes are {self.shape} and {mephisto.shape};")
            return False
        # check matching bins for matmul
        if (self.bins[-1] == mephisto.bins[0]).all():
            if verbose:
                print(str_good)
            return True
        else:
            if raise_err:
                raise ValueError(str_bad + f" bins are {self.bins} and {mephisto.bins}")
            elif verbose:
                print(str_bad + f" bins are {self.bins} and {mephisto.bins}")
            return False

    def match(self, mephisto, verbose=False, raise_err=True) -> bool:
        """Check if two mephistograms are compatible for elementary arithmetics (+, -, *, /).

        Parameters:
        -----------
        mephisto: Mephistogram
            Check if self.histo and Mephistogram.histo have identical dimensions and binnings.
        verbose: bool, default False
            Do or do not print output
        raise_err: bool, default True
            Raise ValueError if mephistograms are not compatible

        Returns:
        --------
        Bool: compatible or not

        """
        str_good = "Elementary arithmetic possible."
        str_bad = "Elementary arithmetic not possible."
        # check if shape is matching
        if not self.shape == mephisto.shape:
            if raise_err:
                raise ValueError(
                    str_bad + f" Shapes are {self.shape} and {mephisto.shape};"
                )
            elif verbose:
                print(str_bad + f" Shapes are {self.shape} and {mephisto.shape};")
            return False

        # check if bins are matching
        if self.ndim == 1:
            same_bins = (self.bins == mephisto.bins).all()
        else:
            same_bins = self.bins == mephisto.bins
        if same_bins:
            if verbose:
                print(str_good)
            return True
        else:
            if raise_err:
                raise ValueError(str_bad + f" bins are {self.bins} and {mephisto.bins}")
            elif verbose:
                print(str_bad + f" bins are {self.bins} and {mephisto.bins}")
            return False

    def T(self):
        """Create and return transposed mephistogram."""
        if self.ndim == 1:
            return self
        elif self.ndim == 2:
            return Mephistogram(self.histo.T, self.bins[::-1], self.axis_names[::-1])
        else:
            raise NotImplementedError("Ö")

    def __add__(self, mephisto):
        """Add two mephistograms."""
        if self.match(mephisto):
            return Mephistogram(self.histo + mephisto.histo, self.bins, self.axis_names)

    def __sub__(self, mephisto):
        """Subtract two mephistograms. Note that the result might have negative numbers."""
        if self.match(mephisto):
            return Mephistogram(self.histo - mephisto.histo, self.bins, self.axis_names)

    def __mul__(self, mephisto):
        """Multiply two mephistograms."""
        if self.match(mephisto):
            return Mephistogram(self.histo * mephisto.histo, self.bins, self.axis_names)

    def __truediv__(self, mephisto):
        """Divide two mephistograms."""
        if self.match(mephisto):
            return Mephistogram(self.histo / mephisto.histo, self.bins, self.axis_names)

    def __matmul__(self, mephisto):
        """Matrix-multiply two mephistograms. -> @ operator

        The resulting binning and axis names will
        correspond to what's expected from matmul:
        First axis: first axis of first bins
        Second axis: second axis of second bins

        """
        if self.match_matmul(mephisto):
            new_bins = (self.bins[0], mephisto.bins[1])
            new_names = (self.axis_names[0], mephisto.axis_names[1])
            return Mephistogram(self.histo @ mephisto.histo, new_bins, new_names)

    def plot(self, **kwargs):
        """Plot 1D or 2D mephistogram.

        **kwargs are piped through to:
        1D: plt.bar
        2D: plt.pcolormesh

        """
        f, axes = plt.subplots(figsize=(5, 4))
        if self.ndim == 2:
            plt.pcolormesh(*self.bins, self.histo.T, **kwargs)
            plt.xlabel(self.axis_names[0])
            plt.ylabel(self.axis_names[1])
            plt.xlim(self.bins[0][0], self.bins[0][-1])
            plt.ylim(self.bins[1][0], self.bins[1][-1])
        elif self.ndim == 1:
            plt.bar(
                get_mids(self.bins),
                height=self.histo,
                width=np.diff(self.bins),
                **kwargs,
            )
            plt.xlabel(self.axis_names)
            plt.xlim(self.bins[0], self.bins[-1])
        else:
            print(f"No plotting possible with {self.ndim} dimensions.")
        return f, axes
