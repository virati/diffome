from typing import List
from diffome.connectome.base import Connectome
from diffome.tda.barcode import BarCode
import matplotlib.pyplot as plt
from gudhi.wasserstein import wasserstein_distance as wass_dist
import numpy as np
from scipy.stats import entropy


class TDAComparison:
    def __init__(self, streams: List[Connectome] = None):
        if streams is None:
            raise ValueError("Streams cannot be None.")
        if len(streams) > 2:
            raise ValueError("Only two streams can be compared.")

        self.streams = streams
        self.first_stack = None
        self.second_stack = None
        self.stacks = [self.first_stack, self.second_stack]

    def calculate(
        self, do_idx=(0, 1), iterations=100, downsample_points=100, print_stats=False
    ):
        first_stack = []
        second_stack = []

        # print out stats here to give an estimate of processing time...
        if print_stats:
            print(
                f"Processing {iterations} iterations with downsampling to {downsample_points} points."
            )

        for ii in range(iterations):
            first_stack.append(
                BarCode(self.streams[do_idx[0]]).calculate(
                    do_plot=False,
                    ignore_streamlines=[ii],
                    downsample_points=downsample_points,
                )
            )
            second_stack.append(
                BarCode(self.streams[do_idx[1]]).calculate(
                    do_plot=False,
                    ignore_streamlines=[ii],
                    downsample_points=downsample_points,
                )
            )

        self.first_stack = first_stack
        self.second_stack = second_stack
        self.stacks = [self.first_stack, self.second_stack]
        self.stack_idxs = do_idx
        self.jack_num = iterations

        return self

    def aggregate_barcodes(self):
        if self.first_stack is None or self.second_stack is None:
            raise ValueError("No barcodes to aggregate. Please run compare() first.")

        first_stack = [
            [
                self.first_stack[iter].barcode[element][1]
                for element in range(len(self.first_stack[iter].barcode))
            ]
            for iter in range(self.jack_num)
        ]
        first_stack = [item for sublist in first_stack for item in sublist]
        second_stack = [
            [
                self.second_stack[iter].barcode[element][1]
                for element in range(len(self.second_stack[iter].barcode))
            ]
            for iter in range(self.jack_num)
        ]
        second_stack = [item for sublist in second_stack for item in sublist]

        self.first_agg = first_stack
        self.second_agg = second_stack

        return self

    def plot_aggregate_barcodes(self, atomic_plots=False):
        for stack in [self.first_agg, self.second_agg]:
            xval = [item[0] for item in stack]
            yval = [item[1] for item in stack]

            plt.scatter(xval, yval, alpha=0.05)
            if atomic_plots:
                plt.show()

        if not atomic_plots:
            plt.title("Aggregate Barcodes")
            plt.xlabel("Birth")
            plt.ylabel("Death")
            plt.legend(["First Connectome", "Second Connectome"])
            plt.show()
        return self

    def calculate_distance_distributions_inside(
        self, which_stack: int = 0, do_plot=False, hold_plot=False
    ):
        if None in self.stacks:
            raise ValueError("No stacks available for distance calculation.")

        self.intra_dist = [] * len(self.stacks)
        for ii in range(len(self.stacks[which_stack])):
            first_barcode = np.array(
                [b for a, b in self.stacks[which_stack][ii].barcode]
            )
            for jj in range(ii + 1, len(self.stacks[which_stack])):
                if ii == jj:
                    continue
                second_barcode = np.array(
                    [b for a, b in self.stacks[which_stack][jj].barcode]
                )
                test = wass_dist(first_barcode, second_barcode)
                self.intra_dist.append(test)

        # calculate KL Divergence for distributions
        # kl_div = entropy(self.intra_dist)
        if do_plot:
            plt.hist(self.intra_dist, bins=30, alpha=0.5)
            plt.title(f"Intra Distance Distribution - Stack {which_stack}")
            plt.xlabel("Distance")
            plt.ylabel("Frequency")

            if not hold_plot:
                plt.show()

        return self

    def calculate_cross_distance(self, do_idx=(0, 1), downsample_factor=50):
        # Calculate w-distance between two connectomes *fully* without subsampling
        first_full_connectome_barcode = (
            BarCode(self.streams[do_idx[0]])
            .calculate(
                do_plot=False,
                ignore_streamlines=[],
                downsample_points=downsample_factor,
            )
            .barcode
        )

        second_full_connectome_barcode = (
            BarCode(self.streams[do_idx[1]])
            .calculate(
                do_plot=False,
                ignore_streamlines=[],
                downsample_points=downsample_factor,
            )
            .barcode
        )

        first_barcode = np.array([b for a, b in first_full_connectome_barcode])
        second_barcode = np.array([b for a, b in second_full_connectome_barcode])

        test = wass_dist(first_barcode, second_barcode)
        print(test)

        return self
