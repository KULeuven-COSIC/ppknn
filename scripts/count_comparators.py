#!/usr/bin/env python3

from math import *

# Compute chunk size
def compute_chunk_size(d, k):
    mu = round(pow(2, ceil(log(k, 2))))
    return (d // 2) + (d % 2) if d <= mu else mu * ceil(d / (2 * mu))

# Count the number of comparators in a merge step
# The parameter nb_unsorted indicates how many wires can be unsorted at the beginning of the output
def count_merge_comparators(d1, d2, k, nb_unsorted=0):
    nb_unsorted = max(min(nb_unsorted, k, d1 + d2), 1)
    d1 = min(d1, k)
    d2 = min(d2, k)
    if d1 * d2 == 0:
        return 0
    elif d1 * d2 == 1:
        return 1
    else:
        size_v = min((d1 // 2) + (d1 % 2) + (d2 // 2) + (d2 % 2), (k // 2) + 1)
        size_w = min((d1 // 2) + (d2 // 2), k // 2)
        return count_merge_comparators((d1 // 2) + (d1 % 2), (d2 // 2) + (d2 % 2), (k // 2) + 1,
                                       (nb_unsorted + 1) // 2) +\
               count_merge_comparators(d1 // 2, d2 // 2, k // 2, ((nb_unsorted - 1) // 2)) +\
               ((size_v + size_w - 1) // 2) - ((nb_unsorted - 1) // 2)

# Count number of comparators in our method
def count_our_comparators(d, k, sorted=False):
    if d == 1:
        return 0
    else:
        chunk_size = compute_chunk_size(d, k)
        return count_our_comparators(chunk_size, k, True) + count_our_comparators(d - chunk_size, k, True) +\
               count_merge_comparators(chunk_size, d - chunk_size, k, 0 if sorted else k)

# Count the number of comparators in the tournament method
def count_tournament_comparators(d, k):
    return (((2 * d) - k - 1) * k) // 2

# Count number of comparators in Yao's method
def count_yao_comparators(d, k):
    if k <= 0 or k >= d:
        return 0
    elif k == 1 or k == d - 1:
        return d - 1
    else:
        return count_yao_comparators(d // 2, k // 2) +\
               count_yao_comparators((d // 2) + (d % 2) + (k // 2), k) + (d // 2)

# Count the number of comparators in the best method that outputs a sorted array
# Note: this is basically a combination of our method and the tournament method
def count_best_sorted_comparators(d, k, sorted_solutions):
    if (d, k) in sorted_solutions:
        return sorted_solutions[(d, k)]

    if d <= 1 or k <= 0:
        sorted_solutions[(d, k)] = 0
    else:
        chunk_size = compute_chunk_size(d, k)
        nb1 = count_best_sorted_comparators(chunk_size, k, sorted_solutions) +\
              count_best_sorted_comparators(d - chunk_size, k, sorted_solutions) +\
              count_merge_comparators(chunk_size, d - chunk_size, k)
        nb2 = (d - 1) + count_best_sorted_comparators(d - 1, k - 1, sorted_solutions)
        sorted_solutions[(d, k)] = min(nb1, nb2)

    return sorted_solutions[(d, k)]

# Count the number of comparators in the best method
def count_best_comparators(d, k, sorted_solutions, best_solutions):
    if (d, k) in best_solutions:
        return best_solutions[(d, k)]

    # This can be achieved via three possible methods:
    # * Our method calling the best sorted method as a subroutine
    # * Tournament method in combination with the best method
    # * Yao's method calling the best method as a subroutine
    if d <= 1 or k <= 0 or k >= d:
        best_solutions[(d, k)] = 0
    else:
        # Our method
        chunk_size = compute_chunk_size(d, k)
        nb1 = count_best_sorted_comparators(chunk_size, k, sorted_solutions) +\
              count_best_sorted_comparators(d - chunk_size, k, sorted_solutions) +\
              count_merge_comparators(chunk_size, d - chunk_size, k, k)

        # Tournament method
        nb2 = (d - 1) + count_best_comparators(d - 1, k - 1, sorted_solutions, best_solutions)

        # Yao's method
        if k == 1 or k == d - 1:
            nb3 = d - 1
        else:
            nb3 = count_best_comparators(d // 2, k // 2, sorted_solutions, best_solutions) +\
                  count_best_comparators((d // 2) + (d % 2) + (k // 2), k, sorted_solutions, best_solutions) + (d // 2)

        best_solutions[(d, k)] = min(nb1, nb2, nb3)

    return best_solutions[(d, k)]



# Print the given comparator to the given file
def print_comparator(file, wire1, wire2):
    file.write(str(min(wire1, wire2)) + ", " + str(max(wire1, wire2)) + "\n")

# Print one merge step of our method to the given file
# The extended set of wires [wires1, wires2] should be sorted (and neither will be modified)
# The parameter nb_unsorted indicates how many wires can be unsorted at the beginning of the output
def print_merge_comparators(file, wires1, wires2, k, nb_unsorted=0):
    nb_unsorted = max(min(nb_unsorted, k, len(wires1) + len(wires2)), 1)
    wires1 = [wires1[index] for index in range(min(len(wires1), k))]
    wires2 = [wires2[index] for index in range(min(len(wires2), k))]
    if len(wires1) == 0 or len(wires2) == 0:
        return
    elif len(wires1) == 1 and len(wires2) == 1:
        print_comparator(file, wires1[0], wires2[0])
    else:
        wires1_even = [wires1[2 * index] for index in range((len(wires1) // 2) + (len(wires1) % 2))]
        wires1_odd = [wires1[2 * index + 1] for index in range(len(wires1) // 2)]
        wires2_even = [wires2[2 * index] for index in range((len(wires2) // 2) + (len(wires2) % 2))]
        wires2_odd = [wires2[2 * index + 1] for index in range(len(wires2) // 2)]
        recursive_size = min(len(wires1_even) + len(wires2_even), (k // 2) + 1) + \
                         min(len(wires1_odd) + len(wires2_odd), k // 2)
        print_merge_comparators(file, wires1_even, wires2_even, (k // 2) + 1, (nb_unsorted + 1) // 2)
        print_merge_comparators(file, wires1_odd, wires2_odd, k // 2, ((nb_unsorted - 1) // 2))
        wires = wires1 + wires2
        for index in range(((nb_unsorted - 1) // 2), (recursive_size - 1) // 2):
            print_comparator(file, wires[2 * index + 1], wires[2 * (index + 1)])

# Print one step of the tournament method to the given file
# The given set of wires should be sorted (and it won't be modified)
# If reverse is set to True, the maximum is computed instead of the minimum
def print_tournament_comparators(file, wires, reverse=False):
    wires = [wires[len(wires) - index - 1 if reverse else index] for index in range(len(wires))]
    while len(wires) > 1:
        for index in range(len(wires) // 2):
            print_comparator(file, wires[index], wires[index + 1])
            wires.pop(index + 1)

# Print one step of Yao's method to the given file
# The given set of wires should be sorted (and it won't be modified)
def print_yao_comparators(file, wires):
    for index in range(len(wires) // 2):
        print_comparator(file, wires[index + (len(wires) % 2)], wires[len(wires) - index - 1])

# Print the top-k network to the given file
# Variable method can be TOURNAMENT, YAO, OUR_SORTED, OUR, BEST_SORTED or BEST
def print_network(file, d, k, sorted_solutions, best_solutions, method="BEST", wires=None):
    assert method == "TOURNAMENT" or method == "YAO" or method == "OUR_SORTED" or method == "OUR" or\
           method == "BEST_SORTED" or method == "BEST"
    if wires is None:
        wires = [index for index in range(d)]

    # This can be achieved via three possible methods:
    # * Our method calling the best sorted method as a subroutine
    # * Tournament method in combination with the best method
    # * Yao's method calling the best method as a subroutine
    if d <= 1 or k <= 0:
        return
    else:
        # Our method
        chunk_size = compute_chunk_size(d, k)
        nb1 = count_best_sorted_comparators(chunk_size, k, sorted_solutions) +\
              count_best_sorted_comparators(d - chunk_size, k, sorted_solutions) +\
              count_merge_comparators(chunk_size, d - chunk_size, k, 0 if method == "BEST_SORTED" else k)

        # Tournament method
        nb2 = (d - 1) + (count_best_sorted_comparators(d - 1, k - 1, sorted_solutions) if method == "BEST_SORTED"
                         else count_best_comparators(d - 1, k - 1, sorted_solutions, best_solutions))

        # Yao's method
        if k >= d:
            nb3 = 0
        elif k == 1 or k == d - 1:
            nb3 = d - 1
        else:
            nb3 = count_best_comparators(d // 2, k // 2, sorted_solutions, best_solutions) +\
                  count_best_comparators((d // 2) + (d % 2) + (k // 2), k, sorted_solutions, best_solutions) + (d // 2)

        # Check which method we will use
        if method == "OUR_SORTED" or method == "OUR" or (method == "BEST_SORTED" and nb1 == min(nb1, nb2)) or\
                                                        (method == "BEST" and nb1 == min(nb1, nb2, nb3)):
            new_method = "OUR_SORTED" if (method == "OUR_SORTED" or method == "OUR") else "BEST_SORTED"
            wires1 = [wires[index] for index in range(chunk_size)]
            wires2 = [wires[chunk_size + index] for index in range(d - chunk_size)]
            print_network(file, chunk_size, k, sorted_solutions, best_solutions, new_method, wires1)
            print_network(file, d - chunk_size, k, sorted_solutions, best_solutions, new_method, wires2)
            print_merge_comparators(file, wires1, wires2, k, 0 if method == "OUR_SORTED" or
                                                                  method == "BEST_SORTED" else k)
        elif method == "TOURNAMENT" or (method == "BEST_SORTED" and nb2 == min(nb1, nb2)) or\
                                       (method == "BEST" and nb2 == min(nb1, nb2, nb3)):
            print_tournament_comparators(file, wires)
            print_network(file, d - 1, k - 1, sorted_solutions, best_solutions, method,
                          [wires[index + 1] for index in range(d - 1)])
        else:
            if k >= d:
                return
            elif k == 1:
                print_tournament_comparators(file, wires)
            elif k == d - 1:
                print_tournament_comparators(file, wires, True)
            else:
                print_yao_comparators(file, wires)
                wires1 = [wires[(d // 2) + (d % 2) + index] for index in range(d // 2)]
                wires2 = [wires[index] for index in range((d // 2) + (d % 2) + (k // 2))]
                print_network(file, d // 2, k // 2, sorted_solutions, best_solutions, method, wires1)
                print_network(file, (d // 2) + (d % 2) + (k // 2), k, sorted_solutions, best_solutions, method, wires2)



# Some tests to find out which method is the best
d = 20
k = 3
print("Yao's method:", count_yao_comparators(d, k))
print("Our method:", count_our_comparators(d, k))
print("Tournament method:", count_tournament_comparators(d, k))

sorted_solutions = dict()
best_solutions = dict()
print("Best method:", count_best_comparators(d, k, sorted_solutions, best_solutions))

with open("../data/network-{}-{}.csv".format(d, k), "w") as file:
    print_network(file, d, k, sorted_solutions, best_solutions)
