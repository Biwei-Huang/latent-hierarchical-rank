from __future__ import annotations

import logging
import pickle
import random
from collections import deque
from copy import deepcopy
from pdb import set_trace
import networkx as nx

import math

import misc as M
from GraphDrawer import printGraph
from Cover import Cover, setLength, getVars, setDifference, setIntersection, deduplicate
from UndirectedGraph import UndirectedGraph
from GraphDrawer import DotGraph
from GaussianGraph import GaussianGraph

LOGGER = logging.getLogger(__name__)

# Class to store discovered latent groups
class LatentGroups:
    def __init__(self, X):
        self.i = 1
        self.X = set([Cover(x) for x in X])
        self.activeSet = set([Cover(x) for x in X])
        self.latentDict = {}
        self.rankDefSets = {}
        self.clusters = {}
        self.nonAtomics = []

    def addRankDefSet(self, Vs, k=1):
        """
        Save a rankDefSet of variables Vs, for merging into clusters later.
        """
        if not k in self.rankDefSets:
            self.rankDefSets[k] = []
        self.rankDefSets[k].append(frozenset(Vs))

    def determineClusters(self):
        """
        From the saved rankDefSets, merge any pair of sets with a common
        element to derive the clusters.
        """
        k = min(list(self.rankDefSets))
        clusters = self.rankDefSets.pop(k)
        n = len(clusters)

        while True:
            i = 0
            j = 1
            while j < len(clusters):
                set1 = clusters[i]
                set2 = clusters[j]

                # Merge overlapping sets
                if len(setIntersection(set1, set2)) > 0:
                    Vs = set1 | set2
                    clusters[i] = Vs
                    clusters.pop(j)

                if j >= len(clusters) - 1:
                    i += 1
                    j = i + 1
                else:
                    j += 1

            if n == len(clusters):
                break
            else:
                n = len(clusters)

        self.clusters[k] = clusters

        # Other rankDefSets of higher cardinality are discarded
        self.rankDefSets = {}

    def confirmClusters(self):
        """
        Given the clusters that we have determined, add each cluster of Vs
        as children of new latent Covers.
        Returns:
            success: boolean - whether any new cluster was successfully added
        """
        k = min(list(self.clusters))
        success = False
        clusters = self.clusters.pop(k)
        for Vs in clusters:
            success = self.addCluster(Vs, k)
        self.clusters = {}
        return success

    def addCluster(self, Vs, k=1):
        """
        For a discovered cluster Vs, create a new latent Cover over it.
        Returns: a boolean indicating whether the addition of a new latent
                 relationship was successful or not (i.e. contradiction)
        """
        parents = self.findParents(Vs)
        parentsSize = setLength(parents)
        gap = k - parentsSize
        LOGGER.info(f"Adding to Dict {Vs} with k={k}")

        # If gap < 0, it means that there is a contradiction, i.e. the
        # current parents of Vs has higher cardinality than the actual rank of
        # testing these Vs together.
        # However, we can just ignore this set of Vs in this case, and
        # hopefully refineClusters will correct the error later on.
        if gap < 0:
            LOGGER.info(
                f"Rejecting {Vs} as a cluster because it is rank {k}"
                f" but has parents of cardinality {parentsSize}"
            )
            return False
        isAtomic = gap > 0

        # Create a new latent Cover, with variables:
        # 1. From existing parents of the cluster Vs
        # 2. Additional new latent variables if required
        newCover = []
        for parent in parents:
            for V in parent.vars:
                newCover.append(V)
        for _ in range(gap):
            newCover.append(f"L{self.i}")
            self.i += 1
        newCover = Cover(newCover, isAtomic)
        LOGGER.info(f"--- Adding {Vs} as a {k}-cluster under {newCover}")

        # Remove children who belong to a subcover
        subcovers = self.findSubcovers(newCover)
        for subcover in subcovers:
            Vs -= self.latentDict[subcover]["children"]

        # Deduplicate cases where Vs includes {L1, {L1, L3}} -> {{L1, L3}}
        Vs = deduplicate(Vs)

        self.addOrUpdateCover(L=newCover, children=Vs)
        return True

    def findParents(self, Vs: set[Cover] | Cover, atomic=False, non_atomic=False):
        """
        Find parents of Vs. Returns empty set if no parents found.

        Args:
            atomic: Whether to take only atomic parents.
            non_atomic: Whether to take only non-atomic parents.
        """
        assert not (
            atomic and non_atomic
        ), "Can only specify atomic or non_atomic, not both."
        parents = set()
        if isinstance(Vs, Cover):
            Vs = set([Vs])

        for parent, values in self.latentDict.items():
            if atomic and not parent.isAtomic:
                continue
            if non_atomic and parent.isAtomic:
                continue
            for V in Vs:
                if V in values["children"]:
                    parents.add(parent)
        parents = deduplicate(parents)

        if non_atomic and len(Vs) == 1:
            assert (
                len(parents) <= 1
            ), f"{next(iter(Vs))} should not have more than one non-atomic parent."
        return parents

    def findAtomicParent(self, L):
        """
        Find the atomic parent of L.

        Raises an error if more than one atomic parent is found, which should
        not be the case.
        Returns None is L is root.
        """
        Ps = self.findParents(L, atomic=True)
        assert len(Ps) <= 1, "Nodes cannot have more than 1 atomic parent."
        P = next(iter(Ps)) if len(Ps) == 1 else None
        return P

    # Check if an AtomicGroup L has observed children
    def hasObservedChildren(self, L):
        for child in self.latentDict[L]["children"]:
            if not child.isLatent:
                return True
        return False

    def updateActiveSet(self):
        """
        Refresh the activeSet after new Covers are added.
        """
        self.activeSet = set()
        # Add all measures to the activeSet
        for X in self.X:
            self.activeSet.add(X)

        # Add all atomic Covers to the activeSet
        # Non-atomic covers are never added, since the atomic covers within
        # would already be in.
        for P in self.latentDict.keys():
            if P.isAtomic:
                self.activeSet.add(P)

        # Remove variables that are children of Covers from activeSet
        for P, val in self.latentDict.items():
            self.activeSet = setDifference(self.activeSet, val["children"])

        self.activeSet = deduplicate(self.activeSet)
        LOGGER.info(f"Active Set: {self.activeSet}")

    def removeCover(self, L: Cover):
        """
        Remove an atomic Cover from the latentDict and activeSet.
        activeSet will be updated at the end to include Children of the Cover.
        """
        assert L.isLatent, "Can only remove latent Cover."
        assert L.isAtomic, "Can only remove atomic Cover."

        # Get the atomicSuperCover of L and remove it
        # e.g. if L=L1 and {L1, L2} is also atomic, we must remove {L1, L2}.
        L = self.findAtomicSuperCover(L)

        # Remove all subsets of L which are also AtomicGroups
        # e.g. {L1, L2} is atomic, and L1 is atomic, so remove both {L1, L2}
        # and L1 from latentDict
        subsets = self.subsets(L)
        for subset in subsets:
            self.latentDict.pop(subset)

        for k in self.latentDict.keys():
            self.latentDict[k]["subcovers"] -= subsets
            self.latentDict[k]["children"] -= subsets

        # L may be a subset of a non-atomic Cover.
        # For this case, we only need to remove the non-atomic Cover, but the
        # other atomic Covers within can remain.
        # E.g. if L=L1 and L2 is also atomic, such that {L1, L2} is non-atomic.
        #      then we only remove {L1, L2} as a Cover but allow L2 to remain.
        nonAtomics, latentDict = self.findNonAtomics(L)
        self.latentDict = latentDict

    def findNonAtomics(self, L):
        """
        Find all nonAtomics associated with L.
        """
        latentDict = {}
        nonAtomics = {}
        for Lp, value in reversed(self.latentDict.items()):
            if not Lp.isAtomic:
                if L.vars < Lp.vars:
                    nonAtomics[Lp] = value
                    continue
            latentDict[Lp] = value
        return nonAtomics, latentDict

    def dissolveNode(self, L):
        """
        Dissolve a latent cover L by:
        1. Making it root
        2. Remove it and L's parent, and add their respective children (in the
           graph where L is root) into the activeSet
        """
        assert isinstance(L, Cover), f"{L} must be Cover"
        assert (
            len(self.activeSet) == 1
        ), f"activeSet is {self.activeSet} but should only have root variable."

        P = self.findAtomicParent(L)
        if P is None:
            # If L root, make another refined node root before continuing
            for V in self.latentDict:
                if V.isAtomic and self.isRefined(V):
                    LOGGER.info(f"{L} is root, making {V} root instead..")
                    self.makeRoot(V)
                    printGraph(self)
                    P = self.findAtomicParent(L)
                    break
        assert (
            P is not None
        ), f"Trying to refine root {L} but no other variable available to set as root."

        # If L is an atomic cover which is a subcover of another atomicCover
        # We should dissolve the larger one instead.
        L = self.findAtomicSuperCover(L)

        LOGGER.info(f"dissolveNode {L}...")

        # Remove L and parent
        printGraph(self)
        LOGGER.info(f"Finding non-atomics for {L}..")
        self.nonAtomics.extend(self.logNonAtomics(L))
        LOGGER.info(f"Finding non-atomics for {P}..")
        self.nonAtomics.extend(self.logNonAtomics(P))
        self.makeRoot(L)
        self.removeCover(L)
        self.removeCover(P)
        self.updateActiveSet()
        M.display(self)
        printGraph(self)
        return True

    # Get all AtomicGroups in a non-AtomicGroup
    def getAtomicsFromGroup(self, Ls):
        assert not Ls.isMinimal(), "Ls must not be minimal"
        groups = set()
        for subcover in self.latentDict[Ls]["subcovers"]:
            if subcover.vars <= Ls.vars:
                groups.add(subcover)
        return groups

    # Make a new connection between parent and child
    def connectNodes(self, parents, children):
        for parent in parents:
            assert parent.isLatent, "Parent must be latent"
            self.addOrUpdateCover(parent, children)

    # Disconnect all linkages between parent and children
    def disconnectNodes(self, parents, children, bidirectional=False):
        for parent in parents:
            self.latentDict[parent]["children"] -= children
        if bidirectional:
            # Remove edges in the other direction as well
            for child in children:
                self.latentDict[child]["children"] -= parents

    # Check if a latent has already been refined
    def isRefined(self, L):
        return self.latentDict[L].get("refined", False)

    # Reduce a list of variable sets by merging them into
    # the minimal set of non-overlapping variable sets
    def mergeList(self, Vlist):
        out = []
        mergeSuccess = False

        while len(Vlist) > 0:
            first = Vlist.pop()
            newVlist = []
            for i, Vs in enumerate(Vlist):
                commonVs = Vs.intersection(first)
                commonVs = [V for V in commonVs if not self.inLatentDict(V)]
                if len(commonVs) > 0:
                    mergeSuccess = True
                    first |= Vs
                else:
                    newVlist.append(Vs)
            Vlist = newVlist
            out.append(first)

        if not mergeSuccess:
            return out
        else:
            return self.mergeList(out)

    def containsCluster(self, Vs):
        """
        Test whether the set of Covers Vs contains any subset such that the
        subset contains > k elements from an existing k-cluster.
        """
        for L, values in self.latentDict.items():
            if L.isAtomic:
                k = len(L)
                children = self.findChildren(L)
                if len(setIntersection(Vs, children)) > k:
                    return True
        return False

    def parentCardinality(self, Vs):
        """
        To compute the cardinality of Vs after we replace any cluster within Vs
        by their latent parents.
        Requires a recursive call as Vs may contain nested clusters.
        """
        Vs = deepcopy(Vs)
        k1 = setLength(Vs)
        for L, _ in self.latentDict.items():
            if L.isAtomic:
                k = len(L)
                children = self.findChildren(L)
                if len(setIntersection(Vs, children)) > k:
                    Vs -= children
                    Vs.add(L)
        k2 = setLength(Vs)
        if k2 < k1:
            return self.parentCardinality(Vs)
        else:
            return k1

    # Check if a variable V already belongs to an AtomicGroup
    def inLatentDict(self, V):
        for _, values in self.latentDict.items():
            if V in values["children"]:
                return True
        return False

    # Given child and parent, reverse their parentage direction
    # i.e. make child the parent instead
    def reverseParentage(self, child, parent):
        assert parent.isLatent, "Parent is not latent"
        assert child.isLatent, "Child is not latent"
        # print(f"Reversing parentage! Parent:{parent} Child:{child}")

        # Remove child as a child of parent
        self.latentDict[parent]["children"] -= set([child])

        # Add parent as a child of child
        self.latentDict[child]["children"].add(parent)

    # Recursive function for use in makeRoot
    def makeRootRecursive(self, Ls, G=None):

        # Make a copy of self, to modify
        if G is None:
            G = deepcopy(self)

        # Parents of L
        # Note: We are finding parents of L based on `self`, not the modified
        #       graph G that is passed around. This is so that we don't end up
        #       in an infinite loop making L -> P then P -> L forever.
        parents = set()
        for L in Ls:
            parents.update(self.findParents(L, atomic=True))

        # If no parents, L is root. Do nothing.
        if len(parents) == 0:
            return G

        # Reverse Direction to parents
        for parent in parents:
            for L in Ls:
                G.reverseParentage(L, parent)
        G = self.makeRootRecursive(parents, G)
        return G

    def makeRoot(self, L: Cover):
        """
        Re-orient latentDict such that L becomes the root node of the graph.

        Note that this procedure does not affect non-atomic Covers.
        """
        assert isinstance(L, Cover), f"{L} must be a Cover."
        G = self.makeRootRecursive(set([L]))
        self.latentDict = G.latentDict
        self.activeSet = set([L])

    # Find all Groups that are a superset of L
    # Including L itself
    def supersets(self, L):
        groups = set()
        for group in self.latentDict:
            if group.vars >= L.vars:
                groups.add(group)
        return groups

    # Find the largest superset for L
    def supersetLargest(self, L):
        k = len(L)
        largest = None
        for group in self.latentDict:
            if group.vars >= L.vars and len(group) >= k:
                largest = group
                k = len(group)
        return largest

    # Find all Groups that are a subset of L
    # Including L itself
    def subsets(self, L):
        groups = set()
        for group in self.latentDict:
            if group.vars <= L.vars:
                groups.add(group)
        return groups

    def findChildren(self, L: Cover, children=set()):
        """
        Recursive search for all immediate children of an atomic Cover
        """
        assert L is not None, "Should not look for None."
        assert L.isAtomic, f"{L} should be an atomic Cover."
        assert L.isLatent, f"{L} should be a latent variable."
        children = children | self.latentDict[L]["children"]
        for subcover in self.latentDict[L]["subcovers"]:
            children = self.findChildren(subcover, children=children)
        return children

    def findNonAtomicChildren(self, L: Cover):
        """
        Find all children of non-Atomic Covers of which L is a subcover.
        """
        children = set()
        for cover in self.latentDict:
            if cover.isAtomic:
                continue
            if L in self.latentDict[cover]["subcovers"]:
                children.update(self.latentDict[cover]["children"])
        return children

    def isRoot(self, L: Cover):
        """
        Check if L is root in that it has no parents.
        """
        parents = self.findParents(L)
        return len(parents) == 0

    def findRandomLatentChild(self, L: Cover):
        children = self.findChildren(L)
        latents = [child for child in children if child.isLatent]
        child = random.sample(latents, k=1)[0]
        return child

    # For a given atomic cover L, find the largest atomicCover of which L is a
    # subset. If L is not a subset of any atomicCover, returns L itself.
    def findAtomicSuperCover(self, L):
        assert isinstance(L, Cover), "L must be a Cover."
        largestCover = L
        superCoverFound = False
        for Lp in self.latentDict:
            if (L.vars < Lp.vars) and Lp.isAtomic:
                largestCover = Lp
                superCoverFound = True
                break
        if superCoverFound:
            return self.findAtomicSuperCover(largestCover)
        else:
            return L

    # Add a child to parents
    def addChildToParents(self, C, Ps):
        latent = C.isLatent
        self.activeSet -= set([C])
        isMinimal = len(Ps) == 1
        parentGroup = Group([V for P in Ps for V in P.vars], isMinimal)

        if parentGroup in self.latentDict:
            self.latentDict[parentGroup]["children"].add(C)
            if latent:
                self.latentDict[parentGroup]["subcovers"].add(C)

        else:
            self.latentDict[parentGroup] = {
                "children": set([C]),
                "subcovers": set([C]) if latent else set(),
            }

    def bypassSingleChild(self, L):
        """
        Given a latent variable L, this function removes the only child of L
        from the graph and connects L to its grandchildren
        """
        children = self.findChildren(L)
        assert len(children) == 1, "Can only perform if single child."
        C = next(iter(children))
        grandchildren = self.findChildren(C)
        self.removeCover(C)
        self.connectNodes(set([L]), grandchildren)  # Connect to grandchild

    def pickAllMeasures(self, Ls):
        visitedA, visitedNA, measures = self._pickAllMeasures(Ls)
        return measures

    def _pickAllMeasures(self, Ls, visitedA=set(), visitedNA=set(), measures=set()):
        """
        Given a set of latent Covers, get all the measured descendants.
        This includes descendants of non-atomic covers.
        """
        visitedA = visitedA.copy()
        visitedNA = visitedNA.copy()
        measures = measures.copy()
        Q = deque()  # FIFO queue for BFS
        for L in Ls:
            if not L.isLatent:
                measures.add(L)
            else:
                Q.append(L)

        # BFS amongst atomic descendants of Ls
        while len(Q) > 0:
            L = Q.popleft()
            visitedA.add(L)
            for C in self.findChildren(L):
                if not C.isLatent:
                    measures.add(C)
                else:
                    Q.append(C)

        # Now check if the visited nodes contain non-atomic covers
        # If yes, do DFS on the children of each non-atomic cover
        for cover in self.latentDict:
            if cover.isAtomic or (cover in visitedNA):
                continue
            if cover.isSubset(visitedA):
                visitedNA.add(cover)
                Cs = set()
                for C in self.latentDict[cover]["children"]:
                    if not C.isLatent:
                        measures.add(C)
                    else:
                        Cs.add(C)
                visitedA, visitedNA, measures = self._pickAllMeasures(
                    Cs, visitedA, visitedNA, measures
                )

        return visitedA, visitedNA, measures

    def saveLatentGroup(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def addOrUpdateCover(self, L: Cover, children: set[Cover] = set()):
        """
        Add a new cover to latentDict with the specified children.
        If cover exists, add the specified children to it.

        Handles the logic of adding subcovers, nonAtomic cover etc.
        """
        subcovers = self.findSubcovers(L)
        L.atomic = not self.isNonAtomic(L)
        if L in self.latentDict:
            self.latentDict[L]["children"].update(children)
            self.latentDict[L]["subcovers"].update(subcovers)
        else:
            self.latentDict[L] = {
                "children": children,
                "subcovers": subcovers,
                "refined": False,
            }

        # If L is a rediscovered non-atomic, we might need to override edge(s)
        # e.g. if we re-discover P1, P2 -> C, but C -> P1, we remove the latter
        if L.atomic:
            return
        for C, v in self.latentDict.items():
            for P in subcovers:
                if (P in v["children"]) and (C in children):
                    LOGGER.info(f"Removing {P} as a child of {C}..")
                    self.latentDict[C]["children"].remove(P)

    def introduceTempRoot(self):
        """
        Add a temporary root over all active variables such that no rank
        deficiency is introduced.

        For n := len(activeSet), we can get rank deficiency of rank k only if
        n >= 2k + 2. So the minimal k to have no rank deficiency is n < 2k + 2,
        i.e. k > n/2 - 1.
        """
        assert len(self.activeSet) > 1, "activeSet should have > 1 Cover."
        LOGGER.info(f"Introducing a temporary root over {self.activeSet}..")
        n = setLength(self.activeSet)
        k = math.ceil(n / 2 - 1 + 0.1)
        self.addCluster(Vs=self.activeSet, k=k)
        self.updateActiveSet()
        assert len(self.activeSet) == 1, "Root should be a single Cover."
        tempRoot = next(iter(self.activeSet))
        tempRoot.temp = True
        self.addOrUpdateCover(tempRoot)

    def connectVariablesChain(self, Vs: set[Cover]):
        """
        Connect variables Vs in a chain structure.
        E.g. Vs={L1, L2, L3, L4}, then:
        - {L1} -> {L2}
        - {L1, L2} -> {L3}
        - {L1, L2, L3} -> {L4}
        """
        Ls = [V for V in Vs if V.isLatent]
        Xs = [X for X in Vs if not X.isLatent]
        j = 1
        while j < len(Ls):
            Ps = set(Ls[0:j])  # Set all but first cover as Parents
            C = Ls[j]
            newCover = Cover(getVars(Ps))
            LOGGER.info(f"Setting {newCover} -----> {C}")
            self.addOrUpdateCover(newCover, children=set([C]))
            j += 1

        # Set any measured variables as children of all latents
        if len(Xs) > 0:
            Ps = set(Ls)
            Cs = set(Xs)
            newCover = Cover(getVars(Ps))
            LOGGER.info(f"Setting {newCover} -----> {Cs}")
            self.addOrUpdateCover(newCover, children=Cs)

    def logNonAtomics(self, L):
        """
        Find nonAtomics associated with L and store their information, namely
        the set of measures that define each atomic cover within the nonAtomic.
        This info will be used to re-identify the nonAtomic variables later.
        """
        nonAtomics, _ = self.findNonAtomics(L)
        infos = []
        for Lp in nonAtomics:
            spouses = []
            LOGGER.info(f"Storing info for non-atomic cover {Lp}..")
            for C in self.latentDict[Lp]["subcovers"]:
                LOGGER.info(f"Storing {C}..")

                # If C is root, we need to set a random child to be root
                # before we record C's measures. Otherwise, C will have all
                # measures recorded which carries no information.
                if self.isRoot(C):
                    child = self.findRandomLatentChild(C)
                    Gp = deepcopy(self)
                    Gp.makeRoot(child)
                    LOGGER.info(f"{C} is root, setting {child} to be root.")
                    measures = Gp.pickAllMeasures(set([C]))
                else:
                    measures = self.pickAllMeasures(set([C]))
                k = len(C)
                spouses.append((k, measures))
                LOGGER.info(f"   For {C}: {measures} measures..")
            infos.append(
                {
                    "spouses": spouses,
                    "children": self.latentDict[Lp]["children"],
                }
            )
        return infos

    def reconnectNonAtomics(self):
        """
        Rediscover the nonAtomic Cover(s) found in self.nonAtomics.
        """

        def _rediscover(nonAtomics=[], retryList=[]):
            """
            Rediscover the nonAtomic Cover corresponding to the item with
            smallest cardinality of measures, and add it back to latentDict.
            """
            # Terminate when queue is empty
            if len(nonAtomics) == 0:
                return retryList

            d, nonAtomics = _findSmallestNonAtomic(nonAtomics)

            # Find the set of covers corresponding to each atomic
            spouses, children = d["spouses"], d["children"]
            failed = False
            discoveredCovers = set()
            for (k, measures) in spouses:
                U = set([next(iter(x.vars)) for x in measures])
                covers = UG.findMinimalSepSet(U, k)
                LOGGER.info(f"Finding Min Sep Set for {U}...")

                # This attempt fails if we fail to find covers for any spouse
                if len(covers) == 0:
                    failed = True
                    retryList.append(d)
                    break

                discoveredCovers.update(covers)
                LOGGER.info(f"   Found {covers} as covers over {U}...")

            # Create the nonAtomic Cover and add it to latentDict
            if not failed:
                coverVars = set()
                for cover in discoveredCovers:
                    coverVars.update(cover.vars)
                nonAtomicCover = Cover(coverVars, atomic=False)
                self.addOrUpdateCover(nonAtomicCover, children)
                LOGGER.info(
                    f"Rediscovered {nonAtomicCover} as a non-atomic"
                    f" parent of {children}.."
                )

            # Continue search with nonAtomics
            nonAtomics = _rediscover(nonAtomics, retryList)
            return retryList

        def _findSmallestNonAtomic(nonAtomics):
            """
            Find the nonAtomicCover with the smallest cardinality of measures.
            We should identify the nonAtomicCover for him first because it is
            easiest to find.
            """
            newlist = []
            lowest = 1e9
            index = None

            # Find smallest cardinality
            for i, d in enumerate(nonAtomics):
                spouses, children = d["spouses"], d["children"]
                card = 0
                for (k, measures) in spouses:
                    card += len(measures)
                if card < lowest:
                    index = i

            # Pop the smallest guy
            for i, v in enumerate(nonAtomics):
                if i != index:
                    newlist.append(v)
            return nonAtomics[index], newlist

        if len(self.nonAtomics) == 0:
            return

        LOGGER.info("Finding nonAtomic Covers...")

        # 1. Create a copy of the graph
        # 2. Fully connect the remaining variables in activeSet
        # 3. Use the UndirectedGraph for rediscovering the nonAtomics
        Gp = deepcopy(self)
        Gp.introduceTempRoot()
        Gp.updateActiveSet()
        UG = UndirectedGraph(Gp)
        self.nonAtomics = _rediscover(self.nonAtomics)

    def findAdjacentNodes(self, L: Cover):
        """
        Find adjacent atomicCovers to L.
        """
        Ns = set()
        Gp = deepcopy(self)
        Gp.makeRoot(L)
        for C in Gp.findChildren(L):
            if C.isTemp:
                Ns.update(Gp.findChildren(C))
            else:
                Ns.add(C)
        return set([V for V in Ns if V.isAtomic])

    def findSubcovers(self, L: Cover, only_atomic=False):
        """
        Find all subcovers of L in latentDict.
        """
        subcovers = set()
        for cover in self.latentDict:
            if only_atomic and not cover.isAtomic:
                continue
            if cover.isSubset(L, strict=True):
                subcovers.add(cover)
        return subcovers

    def isNonAtomic(self, L: Cover):
        """
        Determine if L is nonAtomic, i.e. it can be subdivided into a disjoint
        set of atomic Covers.

        Assumption: no pair of atomic Covers has overlapping variables.
        """
        subcovers = self.findSubcovers(L, only_atomic=True)
        subcoverVars = getVars(subcovers)
        return subcoverVars == L.vars

    def disconnectForNonAtomicParents(self, G: LatentGroups, P: Cover):
        """
        When testing for independence at a child of nonAtomic parent P, we need
        to represent each cover within P with its own disjoint set of variables.
        Hence we need to disconnect the graph at suitable points to achieve this.

        We use BFS to visit descendants of each subcover of P. If we visit the
        same node again, we disconnect all edges to that node.

        Returns:
            a modified LatentGroups graph.
        """
        assert not P.isAtomic, f"{P} should be non atomic."
        Gp = deepcopy(G)
        subcovers = Gp.latentDict[P]["subcovers"]
        visited = set()
        commonNodes = set()

        # First pass to find commonNodes
        for subcover in Gp.latentDict[P]["subcovers"]:
            Gp.makeRoot(subcover)
            Q = deque()
            Q.append(subcover)
            while len(Q) > 0:
                L = Q.pop()
                children = Gp.findChildren(L)
                for child in children:
                    if child in subcovers | visited:
                        commonNodes.add(child)
                    else:
                        if child.isLatent:
                            Q.append(child)
                    visited.add(child)

        # Second pass to disconnect edges
        for subcover in Gp.latentDict[P]["subcovers"]:
            Gp.makeRoot(subcover)
            Q = deque()
            Q.append(subcover)
            while len(Q) > 0:
                L = Q.pop()
                children = Gp.findChildren(L)
                for child in children:
                    if child in subcovers | commonNodes:
                        Gp.disconnectNodes(set([L]), set([child]))
                    else:
                        if child.isLatent:
                            Q.append(child)
        return Gp

    def getDotGraph(self):
        """
        Parse a LatentGroups object into a DotGraph.
        """

        def addParentToGraph(dotGraph, parent, childrenSet):

            # Add edges from children to new parents
            for P in parent.vars:
                for childGroup in childrenSet:
                    for child in childGroup.vars:
                        dotGraph.addEdge(P, child, type=1)

        G = deepcopy(self)
        Xvars = G.X

        # Add X variables
        dotGraph = DotGraph()
        for X in Xvars:
            X = next(iter(X.vars))
            dotGraph.addNode(X)

        # Add nonAtomics first
        for cover in G.latentDict:
            if cover.isAtomic:
                continue
            for L in cover.vars:
                dotGraph.addNode(L, refined=False)

        # Add atomics second so that refined gets reflected correctly
        for cover in G.latentDict:
            if not cover.isAtomic:
                continue
            refined = G.latentDict[cover].get("refined", False)
            for L in cover.vars:
                dotGraph.addNode(L, refined=refined)

        # Work iteratively through the Graph Dictionary
        while len(G.latentDict) > 0:
            parent, values = G.latentDict.popitem()
            addParentToGraph(dotGraph, parent, values["children"])

        return dotGraph

    def pruneControlSet(self, G: LatentGroups, As: set[Cover], Bs: set[Cover]):
        """
        When doing tests for independence, there may exist backdoor connections
        from variables in As to variables in Bs, hence we need to remove those
        variables with backdoor from Bs to prevent messing up the test.

        Returns:
            Bs: A pruned control set.
        """
        Gp = deepcopy(G)
        toPrune = set()
        for A in As:
            visited = set()
            Q = deque()
            Q.append(A)
            while len(Q) > 0:
                V = Q.pop()
                visited.add(V)
                if V in Bs:
                    toPrune.add(V)
                if V.isLatent:
                    for C in Gp.findChildren(V) | Gp.findNonAtomicChildren(V):
                        if not C in visited:
                            Q.append(C)
        return Bs - toPrune

    def disconnectAllEdgestoCover(self, G: LatentGroups, L: Cover):
        """
        Disconnect all edges to L.

        This differs from removeCover in that if L is part of a non-atomic
        cover {L, L2}->C, we want to retain edge from L2->C but remove edge
        from L->C.

        Returns:
            Gp: A modified LatentGroups object
        """
        assert L.isAtomic, f"{L} is not atomic."
        Gp = deepcopy(G)
        for cover, v in G.latentDict.items():

            # Remove L as an atomic parent
            if cover == L:
                Gp.latentDict.pop(L)

            # Remove L as a child of any atomic/non-atomic parent
            if L in v["children"]:
                Gp.latentDict[cover]["children"].remove(L)

            # Remove L as a co-parent, but retain remaining parents
            if not cover.isAtomic:
                if L in v["subcovers"]:
                    subcovers = G.latentDict[cover]["subcovers"] - set([L])
                    newCover = Cover(getVars(subcovers))
                    Gp.addOrUpdateCover(newCover, v["children"])
                    Gp.latentDict.pop(cover)

        return Gp

    def toNetworkX(self):
        """
        Convert graph into a networkx undirected graph.
        """
        NG = nx.Graph()
        for L, v in self.latentDict.items():
            if L.isAtomic:
                for C in v["children"]:
                    NG.add_edge(L, C)
            else:
                for S in v["subcovers"]:
                    for C in v["children"]:
                        NG.add_edge(S, C)
        return NG


def pruneGraph(G: LatentGroups, Vs: set[Cover]):
    """
    Prune away all nodes in the graph G that are descendants of Vs.

    Note that this will result in Vs becoming leaf nodes in the pruned graph.

    Returns:
        Gp: A pruned graph.
    """
    Gp = deepcopy(G)
    for V in Vs:
        assert V.isAtomic, f"{V} is not atomic."

    nodesToDrop = set()
    # BFS to add nodes
    Q = deque()
    for V in Vs:
        Q.append(V)

    while len(Q) > 0:
        A = Q.popleft()
        if A not in Vs:
            nodesToDrop.add(A)
        if A.isLatent:
            for C in Gp.findChildren(A) | Gp.findNonAtomicChildren(A):
                if not C in nodesToDrop | Vs:
                    Q.append(C)

    for node in nodesToDrop:
        if node.isLatent:
            Gp.removeCover(node)
        else:
            for P in Gp.findParents(node):
                if P in Gp.latentDict:
                    Gp.latentDict[P]["children"] -= set([node])

    Gp.X = Gp.X.intersection(Vs)
    Gp.updateActiveSet()
    return Gp


def allRanksEqual(G1: LatentGroups, G2: LatentGroups, Vs: set[Cover]):
    """
    Check that two graphs prescribe the same ranks amongst the set of variables
    in Vs. i.e. for any subset As of Vs and Bs := Vs \ As, rank(As, Bs) is the
    same in G and Gp.
    """
    G1 = deepcopy(G1)
    G1 = pruneGraph(G1, Vs)
    G2 = deepcopy(G2)
    G2 = pruneGraph(G2, Vs)

    # Represent each V with measures if latent
    i = 0
    for V in Vs:
        if V.isLatent:
            Xs = set()
            for j in range(len(V)):
                Xs.add(Cover([f"X_temp{i}"]))
                i += 1
            G1.addOrUpdateCover(V, Xs)
            G2.addOrUpdateCover(V, Xs)

    g1 = getGaussianGraph(G1)
    g2 = getGaussianGraph(G2)
    return M.compareGraphs(g1, g2)[0]


def getGaussianGraph(G: LatentGroups):

    Gp = deepcopy(G)
    g = GaussianGraph()
    assert len(Gp.activeSet) == 1, "ActiveSet should be length 1"

    # Start BFS with root and all non-atomic children
    Q = deque()
    root = next(iter(Gp.activeSet))
    Q.append(root)
    for L in Gp.latentDict:
        if not L.isAtomic:
            for C in Gp.latentDict[L]["children"]:
                Q.append(C)

    while len(Q) > 0:
        V = Q.popleft()

        if not V.isAtomic:
            continue

        parents = Gp.findParents(V)

        # Root var
        if len(parents) == 0:
            for v in V.vars:
                g.add_variable(v, None)

        # Check if all parents are in g, otherwise pass first
        else:
            parentVars = getVars(parents)
            if parentVars <= set(g.vars):
                for v in V.vars:
                    g.add_variable(v, parentVars)
            else:
                Q.append(V)
                continue

        # Continue search
        try:
            if V.isLatent and V.isAtomic:
                for C in Gp.findChildren(V):
                    Q.append(C)
        except:
            set_trace()
    return g
