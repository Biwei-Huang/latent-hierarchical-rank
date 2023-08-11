"""
A class to perform the cross cover test.
"""

from copy import deepcopy
from Cover import Cover
from LatentGroups import LatentGroups
from pdb import set_trace


class CrossCoverTester:
    def __init__(self, Vs, A, B) -> None:
        """
        A, B: The two latent variables we want to test for cond. independence.
        Vs: A dictionary where the key is each latent variable for testing
        """
        self.A = A
        self.B = B
        self.Vs = Vs

        # Compute expected rank (see getExpectedRank)
        self.expectedRank = sum([len(V) for V in Vs])

        # Get list of children sorted by decreasing cardinality
        self.childList = []
        for V in Vs.values():
            for group in V:
                self.childList.append(group)
        self.childList.sort(key=lambda x: -len(x))

        # Reverse Dict from child to parent
        self.reverseDict = {}
        for k, v in self.Vs.items():
            for group in v:
                self.reverseDict[group] = k

    def getSetCardinality(self, setA):
        """
        Calculate ||set|| as the cardinality of the children in each set, capped
        at the cardinality of the latent variable
        """
        tempDict = {}  # Store cardinality for each latent in this dict
        for V in setA:
            parent = self.reverseDict[V]
            if parent in tempDict:
                tempDict[parent] = min(tempDict[parent] + len(V), len(parent))
            else:
                tempDict[parent] = min(len(V), len(parent))
        return sum([x for x in tempDict.values()])

    def getMinCardinality(self, setA, setB):
        """
        Calculate the expected rank of setA and setB respectively, and take
        the smaller quantity. This is the expected rank of the subcovariance
        if no rank deficiency is found.
        """
        cardA = self.getSetCardinality(setA) + len(self.A)
        cardB = self.getSetCardinality(setB) + len(self.B)
        return min(cardA, cardB)

    def getExpectedRank(self):
        """
        The expected rank is the rank of the subcovariance matrix supposing that
        the latent variables in Vs do indeed d-separate A from B.
        In other words, the expected rank is the sum of the size of the latents
        in Vs, as they are sufficient to d-separate setA from setB.
        """
        return self.expectedRank

    def addToSet(self, setA, setB, childList, V, choice="A") -> None:
        """
        Add a V from Vs into the choice set.
        If the addition of V causes the cardinality of the variables in the set
        to exceed the cardinality of L=parent(V), we add all other children of L
        into the other set, since further adding to this set will not help to
        improve cardinality. This is to help speed up search.
        """

        # Make a copy
        setA, setB, childList = [deepcopy(x) for x in [setA, setB, childList]]

        if choice == "A":
            choiceSet, otherSet = setA, setB
        elif choice == "B":
            choiceSet, otherSet = setB, setA
        else:
            raise ValueError("Must be A or B.")

        # Add V to choiceSet
        choiceSet.add(V)
        parent = self.reverseDict[V]

        # Compute the cardinality of all children of parent in choiceSet
        cardinality = 0
        for V in choiceSet:
            if V in self.Vs[parent]:
                cardinality += len(V)

        # If saturation occurs, add remaining children of parent into otherSet
        # Also remove them from childList
        if cardinality >= len(parent):
            for child in self.Vs[parent]:
                if child in childList:
                    otherSet.add(child)
                    childList.remove(child)
        return setA, setB, childList

    def search(self):
        """
        Search for a setA, setB such that it has higher cardinality than the
        expectedRank, which means that it is suitable for testing.
        Main function of interest.
        Uses a binary tree with Complete Greedy Algorithm to find them.
        """
        childList = deepcopy(self.childList)
        setA, setB, cardinality = self._search(set(), set(), childList)
        satisfied = cardinality > self.expectedRank
        setA = set([self.A]) | setA
        setB = set([self.B]) | setB
        return setA, setB, cardinality, satisfied

    def _search(self, setA, setB, childList):
        """
        Recursive function
        """

        # Terminate
        if len(childList) == 0:
            cardinality = self.getMinCardinality(setA, setB)
            return setA, setB, cardinality

        # Branching
        child = childList.pop()

        # Check which branch is more promising
        Acard = self.getMinCardinality(setA | set([child]), setB)
        Bcard = self.getMinCardinality(setA, setB | set([child]))

        # First branch to more promising side
        firstChoice = "A" if (Acard >= Bcard) else "B"
        setA1, setB1, childList1 = self.addToSet(
            setA, setB, childList, child, firstChoice
        )
        setA1, setB1, cardinality1 = self._search(setA1, setB1, childList1)

        # Early Return if first branch works
        if cardinality1 > self.getExpectedRank():
            return setA1, setB1, cardinality1

        # Otherwise, check less promising side
        nextChoice = "B" if (Acard >= Bcard) else "A"
        setA2, setB2, childList2 = self.addToSet(
            setA, setB, childList, child, nextChoice
        )
        setA2, setB2, cardinality2 = self._search(setA2, setB2, childList2)

        # Finally return the side with higher cardinality
        if cardinality1 >= cardinality2:
            return setA1, setB1, cardinality1
        else:
            return setA2, setB2, cardinality2
