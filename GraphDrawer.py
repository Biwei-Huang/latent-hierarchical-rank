import pydot
import json


PARAMS = {
    "default_node_colour": "black",
    "refined_node_colour": "red",
    "default_edge_colour": "black",
    "directed_edge_colour": "black",
}
with open("params.json") as f:
    PARAMS.update(json.load(f))


class DotGraph:
    """
    A class used to construct a directed/undirected graph and parses the graph
    into a .dot file for pydot plotting.
    """

    def __init__(
        self,
        default_node_colour: str = PARAMS["default_node_colour"],
        refined_node_colour: str = PARAMS["refined_node_colour"],
        default_edge_colour: str = PARAMS["default_edge_colour"],
        directed_edge_colour: str = PARAMS["directed_edge_colour"],
    ):
        self.default_node_colour = default_node_colour
        self.refined_node_colour = refined_node_colour
        self.default_edge_colour = default_edge_colour
        self.directed_edge_colour = directed_edge_colour
        self.nodes = set()
        self.dirEdges = set()
        self.undirEdges = set()
        self.nodecolor = {}

    def addNode(self, V, refined=False):
        self.nodes.add(V)
        if refined:
            self.nodecolor[V] = self.refined_node_colour
        else:
            self.nodecolor[V] = self.default_node_colour

    # 0 for undirected, 1 for directed
    def addEdge(self, u, v, type=0):
        if type == 0:
            self.undirEdges.add(frozenset([u, v]))

        elif type == 1:
            self.dirEdges.add((u, v))

    def edges(self, V, type=0):
        edgelist = []
        if type == 0:
            for edge in self.undirEdges:
                if V in edge:
                    edgelist.append(edge)

        if type == 1:
            for edge in self.dirEdges:
                if V in edge:
                    edgelist.append(edge)
        return edgelist

    def removeUndirEdgesFromNode(self, V):
        edgesToRemove = set()
        for edgeSet in self.undirEdges:
            if V in edgeSet:
                edgesToRemove.add(edgeSet)
        self.undirEdges = self.undirEdges - edgesToRemove

    def toDot(self, outpath: str):
        # TODO: Improve plotting for phase III with undirected edges
        text = "digraph {\n"

        # Add nodes
        for node in self.nodes:
            text += f"{node} [color = {self.nodecolor[node]}]; "
        text += "\n"

        # Add undirected edges
        text += "subgraph Undirected {\n"
        text += f"edge [dir=none, color={self.default_edge_colour}]\n"
        for edgeSet in self.undirEdges:
            edgeSet = list(edgeSet)
            text += f"{edgeSet[0]} -> {edgeSet[1]}\n"

        text += "}\n\n"

        # Add directed Edges
        text += "subgraph Directed {\n"
        text += f"edge [color={self.directed_edge_colour}]\n"
        for edgeSet in self.dirEdges:
            edgeSet = list(edgeSet)
            text += f"{edgeSet[0]} -> {edgeSet[1]}\n"

        text += "}\n\n"
        text += "}\n"
        with open(outpath, "w") as f:
            f.write(text)


def printGraph(O: object, outpath="plots/test.png", layout="dot", res=25):
    """
    Function to plot a graph object using pydot from various types of graph
    objects.
    """
    if isinstance(O, DotGraph):
        dotGraph = O
    else:
        try:
            dotGraph = O.getDotGraph()
        except:
            raise TypeError(f"{O} is of type {type(O)}, not supported.")
    dotGraph.toDot("temp.dot")
    graphs = pydot.graph_from_dot_file("temp.dot")
    graphs[0].set_size(f'"{res},{res}!"')
    graphs[0].set_layout(layout)
    graphs[0].write_png(outpath)
