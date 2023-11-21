#!/usr/bin/env python3

import matplotlib.pyplot as plt
import count_comparators as cc

DS = [100, 200, 300, 400, 500, 600, 700, 800]

def zuber(d, k):
    return ((d**2) - d)/2. + (d**2)/(65. - k)

def ours(d, k):
    best_solutions = dict()
    return cc.count_best_comparators(d, k, best_solutions)


if __name__ == "__main__":
    reds = (1, 0.1, 0.1)
    blues = (0.1, 0.1, 1)

    for i, k in enumerate([5, 10, 15, 20]):
        plt.plot(DS, [zuber(d, k) for d in DS], '--', label='ZS21, $k={}$'.format(k), color=reds+(1-0.2*i,))
    for i, k in enumerate([5, 10, 15, 20]):
        plt.plot(DS, [ours(d, k) for d in DS], ':', label='Ours, $k={}$'.format(k), color=blues+(1-0.2*i,))

    plt.legend(loc="upper left")
    plt.yscale('log')
    plt.xlabel('$d$')
    plt.ylabel('Number of PBS (log scale)')
    plt.grid()
    # plt.show()
    plt.savefig('plot.pdf')
