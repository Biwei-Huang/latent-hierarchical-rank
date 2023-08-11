"""
A simple implementation of an undirected tree graph.
"""
from __future__ import annotations
from Cover import Cover
import logging

LOGGER = logging.getLogger(__name__)


class UndirectedGraph:
    def __init__(self, G=None, test=None):
        if not test is None:
            G = test()
        self.nodes, self.edges = self.makeUndirectedGraph(G)
        self.measures = {node for node, v in self.nodes.items() if not v["isLatent"]}

    def makeUndirectedGraph(self, G: LatentGroups):
        """
        Given a latentDict from a LatentGroups object which represents a
        directed graph, represent it as an undirected graph where each
        atomicCover forms a node.
        Note that we do not consider edges from non-atomic covers, as the
        intention is to simply find the corresponding atomic covers to each
        set of measures and non-atomics do not help.
        """
        edges = {}
        nodes = {}

        # Add all nodes
        measures = [next(iter(x.vars)) for x in G.X]
        for measure in measures:
            nodes[measure] = {"size": 1, "isLatent": False}

        for cover in G.latentDict:
            if not cover.isAtomic:
                continue

            coverName = self.getCoverName(cover)
            k = len(cover)
            nodes[coverName] = {"size": k, "isLatent": True, "cover": cover}

        # Add all edges
        edges = {node: set() for node in nodes}
        for cover in G.latentDict:
            if not cover.isAtomic:
                continue
            coverName = self.getCoverName(cover)

            # Add children
            children = G.latentDict[cover]["children"]
            for child in children:
                childName = self.getCoverName(child)
                edges[coverName].add(childName)
                edges[childName].add(coverName)

            # Add subcovers (subcovers are treated as a separate cover)
            # E.g. Suppose we have:
            #   {L1}: X1, X2 | isAtomic=True
            #   {L1, L2}: X3, X4 | isAtomic=True, subcovers={L1}
            # We represent this as a graph:
            #   "L1,L2" --- "L1", "X3", "X4"
            #   "L1" --- "X1", "X2"
            subcovers = G.latentDict[cover]["subcovers"]
            for subcover in subcovers:
                subcoverName = self.getCoverName(subcover)
                edges[coverName].add(subcoverName)
                edges[subcoverName].add(coverName)

        return nodes, edges

    def findNeighbours(self, nodes: str | set[str]):
        """
        Return a set of the node's adjacent neighbours.
        """
        if isinstance(nodes, str):
            nodes = set([nodes])
        neighbours = set()
        for node in nodes:
            neighbours |= self.edges[node]
        neighbours = neighbours - nodes
        return neighbours

    def findMinimalSepSet(self, U: set[str], k: int):
        """
        For a set of measures U, find the minimal separating set of nodes S
        that is nearest to U and separates U from all other measures, and
        ||S|| = k.
        Returns an empty set if no such S exists.
        """

        def _findSepSet(C: set[str], V: set[str], visited=set(), S=set(), T=set()):
            """
            For a candidate separating set C, check if any node in C is redundant
            in that it cannot be reached by any node in V.
            Return the final non-redundant subset of C (i.e. S).
            """
            newV = set()
            for v in V:
                visited.add(v)
                B = self.findNeighbours(v)
                for b in B:
                    # If b is already visited, stop search
                    if b in visited:
                        continue

                    # If b is in candidate set, stop search and record
                    if b in C:
                        S.add(b)
                        T.add(v)

                    # Otherwise, continue search from b
                    else:
                        newV.add(b)

            # Terminate if queue is empty
            if len(newV) == 0:
                return visited, S, T

            visited, S, T = _findSepSet(C, newV, visited, S, T)
            return visited, S, T

        def _step(V: set[str], S: set[str], T: set[str], k: int):
            """
            For a given separating set S, try moving S further away from U and
            check if the new separating set is of smaller cardinality.
            Recursively step until no smaller S can be found.
            """
            # Early terminate if cardinality k is reached
            if self.cardinality(S) == k:
                return S, T

            improved = False
            for s in S:
                C = (S - set([s])) | (self.findNeighbours(s).intersection(T))
                _, Snew, Tnew = _findSepSet(C, V)
                if self.cardinality(Snew) < self.cardinality(S):
                    S, T = Snew.copy(), Tnew.copy()
                    improved = True
                    break
            if improved:
                S, T = _step(V, S, T)
            return S, T

        assert all([u.startswith("X") for u in U]), "U must be all measures."
        V = self.measures - U
        C = self.findNeighbours(U)
        _, S, T = _findSepSet(C, V)
        S, T = _step(V, S, T, k=k)

        # Reject separating set if ||S|| > k
        if self.cardinality(S) > k:
            return set()

        covers = set([self.nodes[node]["cover"] for node in S])
        return covers

    def getCoverName(self, cover):
        return ",".join(sorted(cover.vars))

    def cardinality(self, S):
        card = 0
        for s in S:
            card += len(s.split(","))
        return card


# For Testing
def addOrUpdate(G, covers, children, subcovers=[], isAtomic=True):
    if isinstance(covers, str):
        covers = [covers]

    children2 = []
    for C in children:
        if isinstance(C, str):
            children2.append(Cover([C]))
        else:
            children2.append(Cover(C))

    subcovers2 = []
    for C in subcovers:
        if isinstance(C, str):
            subcovers2.append(Cover([C]))
        else:
            subcovers2.append(Cover(C))

    G.addOrUpdateCover(Cover(covers, isAtomic), set(children2), set(subcovers2))


def scenario3a():
    from LatentGroups import LatentGroups

    # Example 3a
    G = LatentGroups([f"X{i}" for i in range(1, 12)])
    addOrUpdate(G, "L1", ["L2", "L3", "X10"])
    addOrUpdate(G, "L2", ["X9", ["L4", "L5"]])
    addOrUpdate(G, "L3", ["X11", ["L6", "L7"]])
    addOrUpdate(G, ["L4", "L5"], ["X1", "X2", "X3", "X4"])
    addOrUpdate(G, ["L6", "L7"], ["X5", "X6", "X7", "X8"])
    return G


def scenario1b():
    from LatentGroups import LatentGroups

    G = LatentGroups([f"X{i}" for i in range(0, 13)])
    addOrUpdate(G, "L7", ["L1", "L3", "X4"])
    addOrUpdate(G, ["L6", "L7"], [["L4", "L5"]], ["L6", "L7"], False)
    addOrUpdate(G, "L6", ["X5", "L2"])
    addOrUpdate(G, ["L2", "L3"], ["X12"], ["L2", "L3"], False)
    addOrUpdate(G, ["L4", "L5"], ["X9", "X6", "X7", "X8"])
    addOrUpdate(G, "L3", ["X10", "X11"])
    addOrUpdate(G, "L2", ["X2", "X3"])
    addOrUpdate(G, "L1", ["L6", "X0", "X1"])
    G.activeSet = set([Cover("L7")])
    G.i = 8
    return G


if __name__ == "__main__":
    UG = UndirectedGraph(test=scenario1b)
    U = set(["X0", "X1", "X4", "X10", "X11", "X5"])
    S = UG.findMinimalSepSet(U, k=1)
    LOGGER.info(S)
