import numpy as np
import matplotlib.pyplot as plt


def get_mids(bins):
    """Calculate the bin mids from an array of bins"""
    return (bins[1:] + bins[:-1]) / 2


def rebin(histo, bins):
    """TODO: update histogram to different binning via RegularGridInterpolator"""
    pass


def like(mephisto, fill_value=0):
    """Make new Mephistogram with generic fill value,
    but otherwise same properties"""
    return Mephistogram(
        np.full_like(mephisto.histo, fill_value=fill_value),
        bins=mephisto.bins,
        axis_names=mephisto.axis_names,
        make_hist=False,
    )


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

    def __len__(self):
        return len(self.histo)

    def __getitem__(self, idx):
        return self.histo[idx]

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
        rep_str += f" Axis names are {self.axis_names}."
        rep_str += "\n" + str(self.histo)
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

    def normalize(self, mode="1D", axis=0):
        """Normalize histogram.
        Parameters:
        -----------
        mode: '1D' or 'full'
        axis: optional, only needed if mode='1D'; 0 or 1
        """
        self.histo = self.histo.astype("float")
        if mode == "1D":
            if axis == 0:
                self.histo /= np.sum(self.histo, axis=axis)
            elif axis == 1:
                self.histo /= np.sum(self.histo, axis=axis)[:, np.newaxis]
            else:
                raise ValueError("Too many axis dimensions")
        elif mode == "full":
            self.histo /= np.sum(self.histo)
        else:
            raise ValueError("Unknown mode:", mode, ". Should be 1D or full!")

    def sum(self, return_mephisto=False, **kwargs):
        """Wrapper for numpy.sum;

        Parameters:
        -----------
        return_mephisto: bool, default False
            Option to return Mephistogram if original Mephistogram was >= 2D.

        kwargs:
        -------
        All piped through to numpy.sum(**kwargs)

        """
        # return array/number
        if self.ndim == 1 or not return_mephisto or not "axis" in kwargs.keys():
            return self.histo.sum(**kwargs)
        else:
            axis = kwargs["axis"]
            if self.ndim == 2:
                remaining_axis = 1 if axis == 0 else 0
                return Mephistogram(
                    self.histo.sum(**kwargs),
                    bins=self.bins[remaining_axis],
                    axis_names=self.axis_names[remaining_axis],
                )
            else:
                remaining_axes = [i for i in range(self.ndim) if i != axis]
                return Mephistogram(
                    self.histo.sum(**kwargs),
                    bins=tuple(self.bins[r] for r in remaining_axes),
                    axis_names=tuple(self.axis_names[r] for r in remaining_axes),
                )

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
            same_bins = True
            for b, mb in zip(self.bins, mephisto.bins):
                same_bins &= (b == mb).all()
                if not same_bins:
                    break
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

    # Logic
    def __and__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.hist & this

    def __rand__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return this & self.hist

    def __or__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.hist | this

    def __ror__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return this | self.hist

    # Comparisons
    def __eq__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.histo == this

    def __ne__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.histo != this

    def __ge__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.histo >= this

    def __gt__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.histo > this

    def __le__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.histo <= this

    def __lt__(self, this: object) -> bool:
        if isinstance(this, Mephistogram):
            self.match(this)
        return self.histo < this

    # Elementary arithmetics.
    def __neg__(self):
        return self * -1

    def inv(self, in_place=False):
        """Inverting every histo element of mephistogram. NOT a matrix inversion!"""
        if in_place:
            self.histo = 1 / self.histo
        else:
            return Mephistogram(1 / self.histo, self.bins, self.axis_names)

    def __add__(self, this):
        """Add two mephistograms or a number/matching array."""
        if (
            isinstance(this, int)
            or isinstance(this, float)
            or isinstance(this, np.ndarray)
        ):
            return Mephistogram(self.histo + this, self.bins, self.axis_names)
        elif isinstance(this, Mephistogram) and self.match(this):
            return Mephistogram(self.histo + this.histo, self.bins, self.axis_names)
        else:
            raise TypeError(f"Operation not defined for this {type(this)}")

    def __radd__(self, this):
        return self + this

    def __sub__(self, this):
        """Subtract two mephistograms or a number/matching array. Note that the result might have negative numbers."""
        if (
            isinstance(this, int)
            or isinstance(this, float)
            or isinstance(this, np.ndarray)
        ):
            return Mephistogram(self.histo - this, self.bins, self.axis_names)
        elif isinstance(this, Mephistogram) and self.match(this):
            return Mephistogram(self.histo - this.histo, self.bins, self.axis_names)
        else:
            raise TypeError(f"Operation not defined for this {type(this)}")

    def __rsub__(self, this):
        return -self + this

    def __mul__(self, this):
        """Multiply two mephistograms or a number/matching array."""
        if (
            isinstance(this, int)
            or isinstance(this, float)
            or isinstance(this, np.ndarray)
        ):
            return Mephistogram(self.histo * this, self.bins, self.axis_names)
        elif isinstance(this, Mephistogram) and self.match(this):
            return Mephistogram(self.histo * this.histo, self.bins, self.axis_names)
        else:
            raise TypeError(f"Operation not defined for this {type(this)}")

    def __rmul__(self, this):
        return self.__mul__(this)

    def __truediv__(self, this):
        """Divide two mephistograms or a number/matching array."""
        if (
            isinstance(this, int)
            or isinstance(this, float)
            or isinstance(this, np.ndarray)
        ):
            return Mephistogram(self.histo / this, self.bins, self.axis_names)
        elif isinstance(this, Mephistogram) and self.match(this):
            return Mephistogram(self.histo / this.histo, self.bins, self.axis_names)
        else:
            raise TypeError(f"Operation not defined for this {type(this)}")

    def __rtruediv__(self, this):
        return self.inv() * this

    def __matmul__(self, this):
        """Matrix-multiply two mephistograms. -> @ operator

        The resulting binning and axis names will
        correspond to what's expected from matmul:
        First axis: first axis of first bins
        Second axis: second axis of second bins

        """
        # in contrast to the other operations,
        # we want to be really sure that these are both mephistograms
        if isinstance(this, Mephistogram) and self.match_matmul(this):
            new_bins = (self.bins[0], this.bins[1])
            new_names = (self.axis_names[0], this.axis_names[1])
            return Mephistogram(self.histo @ this.histo, new_bins, new_names)
        else:
            raise TypeError(f"Operation not defined for this {type(this)}")

    def plot(self, **kwargs):
        """Wrapper."""
        plot_mephistogram(self, **kwargs)


def plot_multiple_mephistograms(mephistograms, **kwargs):
    """Plot multiple 1D mephistograms as barstacked."""
    if not "f" in kwargs or not "axes" in kwargs:
        f, axes = plt.subplots(figsize=(5, 4))
    else:
        f = kwargs.pop("f")
        axes = kwargs.pop("axes")
    labels = kwargs.pop("labels", [""] * len(mephistograms))
    for ii, mephistogram in enumerate(mephistograms):
        # check if mephistogram matches the previous one
        if ii > 0:
            mephistogram.match(mephistograms[ii - 1], raise_err=True)
        plt.bar(
            mephistogram.bin_mids,
            height=mephistogram.histo,
            width=np.diff(mephistogram.bins),
            bottom=0 if ii == 0 else mephistograms[ii - 1].histo,
            label=labels[ii],
            **kwargs,
        )
    plt.xlabel(mephistogram.axis_names)
    plt.xlim(mephistogram.bins[0], mephistogram.bins[-1])
    return f, axes


def plot_mephistogram(mephistogram, **kwargs):
    """Plot 1D or 2D mephistograms.

    **kwargs are piped through to:
    1D: plt.bar
    2D: plt.pcolormesh

    """
    if not "f" in kwargs or not "ax" in kwargs:
        f, ax = plt.subplots(figsize=(5, 4))
    else:
        f = kwargs.pop("f")
        ax = kwargs.pop("ax")
    if mephistogram.ndim == 2:
        im = ax.pcolormesh(*mephistogram.bins, mephistogram.histo.T, **kwargs)
        ax.set_xlabel(mephistogram.axis_names[0])
        ax.set_ylabel(mephistogram.axis_names[1])
        ax.set_xlim(mephistogram.bins[0][0], mephistogram.bins[0][-1])
        ax.set_ylim(mephistogram.bins[1][0], mephistogram.bins[1][-1])
        f.colorbar(im)
    elif mephistogram.ndim == 1:
        ax.bar(
            mephistogram.bin_mids,
            height=mephistogram.histo,
            width=np.diff(mephistogram.bins),
            **kwargs,
        )
        ax.set_xlabel(mephistogram.axis_names)
        ax.set_xlim(mephistogram.bins[0], mephistogram.bins[-1])
    else:
        print(f"No plotting possible with {mephistogram.ndim} dimensions.")
    return f, ax
