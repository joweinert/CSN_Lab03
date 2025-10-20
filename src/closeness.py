import networkx as nx
import rustworkx as rx
from datetime import datetime, timedelta

from src.graph_extraction import load_edges


def nx_compute_closeness_centrality(edges, benchmark: bool = False):
    """
    Compute the closeness centrality for each node given an edge list using NetworkX.
    """
    if benchmark:
        start = datetime.now()

    G = nx.Graph()
    G.add_edges_from(edges)

    nx_closeness = nx.closeness_centrality(G)

    if benchmark:
        end = datetime.now()
        duration = end - start
        print(f"NetworkX closeness centrality computed in {duration}")

    return nx_closeness


def rx_compute_closeness_centrality(edges, benchmark: bool = False):
    """
    Compute the closeness centrality for each node given an edge list using Rustworkx.
    """
    if benchmark:
        start = datetime.now()

    G = rx.PyGraph(multigraph=False)
    G.extend_from_edge_list(edges)

    rustworkx_closeness = rx.closeness_centrality(G)

    if benchmark:
        end = datetime.now()
        duration = end - start
        print(f"Rustworkx closeness centrality computed in {duration}")

    return rustworkx_closeness


if __name__ == "__main__":
    edges = load_edges(lang="en", int_optimize=True)
    nx_closeness = nx_compute_closeness_centrality(edges, benchmark=True)
    rx_closeness = rx_compute_closeness_centrality(edges, benchmark=True)

    def print_result(closeness_dict, library_name):
        print(f"\n{library_name} closeness centrality results:")
        for i, (node, centrality) in enumerate(
            sorted(closeness_dict.items(), key=lambda x: x[0])
        ):
            if i >= 10:
                break
            print(f"Node: {node}, Closeness Centrality: {centrality}")

    print_result(nx_closeness, "NetworkX")
    print_result(rx_closeness, "Rustworkx")
