import rustworkx as rx
import networkx as nx
import random
from itertools import combinations


def erdos_renyi_null_model_rx(N, E, seed=None) -> rx.PyGraph:
    """https://www.rustworkx.org/apiref/rustworkx.undirected_gnm_random_graph.html

    Generate an undirected Erdős-Rényi graph with N nodes and E edges."""
    return rx.undirected_gnm_random_graph(N, E, seed=seed)


def erdos_renyi_null_model_nx(N, E, seed=None) -> nx.Graph:
    """Generate an undirected Erdős-Rényi graph with N nodes and E edges.

    RUSTWORKX VERSION IS RECOMMENDED FOR PERFORMANCE.
    """
    return nx.gnm_random_graph(N, E, seed=seed, directed=False)


def edge_switching_null_model(edges: list[tuple[int, int]], N, E, Q=20, seed=None):
    assert len(edges) == E, "Number of edges does not match E parameter"
    rng = random.Random(seed)

    edgelist = [tuple(sorted(e)) for e in edges]
    edge_set = set(edgelist)  # faster duplicate checks (hash -> O(1) vs O(n))

    trials = Q * E
    for _ in range(trials):
        i = rng.randrange(E)
        j = rng.randrange(E - 1)
        # pls two different edges (actually a nice trick to achieve this unbiased:)
        if j >= i:
            j += 1

        (u, v) = edgelist[i]  # undirected, sooo u,v == v,u
        (x, y) = edgelist[j]

        # distinct nodes. Otherweise we would either produce a self-loop or no change
        if len({u, v, x, y}) < 4:
            continue

        possible_swaps = [((u, x), (v, y)), ((u, y), (v, x))]
        (a, b), (c, d) = rng.choice(possible_swaps)
        (a, b), (c, d) = tuple(sorted((a, b))), tuple(sorted((c, d)))

        # no multigraph!
        if (a, b) in edge_set or (c, d) in edge_set:
            continue

        edge_set.remove((u, v))
        edge_set.remove((x, y))
        edge_set.add((a, b))
        edge_set.add((c, d))

        # suing the index speeds things up -> replace instead of removal+append
        edgelist[i] = (a, b)
        edgelist[j] = (c, d)

    G = rx.PyGraph(multigraph=False, node_count_hint=N, edge_count_hint=E)
    G.extend_from_edge_list(edgelist)

    return G
