from CrossCoverTester import CrossCoverTester
from Group import Group
from pdb import set_trace

Vs = {
    Group(["L1"]): set([Group(x) for x in ["X1", "X2"]]),
    Group(["L3", "L4"]): set([Group(x) for x in ["X5", "X6", "X7"]]),
}

setA, setB, cardinality, satisfied = CrossCoverTester(
    Vs, Group("L2"), Group(["L5", "L6"])
).search()
set_trace()
