from collections import defaultdict
import math
import rustworkx as rx
import networkx as nx
import joblib
from contextlib import nullcontext
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import random
import numpy as np

from src.utils import bcolors
from src.null_model import (
    erdos_renyi_null_model_rx,
    erdos_renyi_null_model_nx,
    edge_switching_null_model,
)

NULL_MODELS = {
    "ER": erdos_renyi_null_model_rx,
    "ER_NX": erdos_renyi_null_model_nx,
    "ES": edge_switching_null_model,
}


# By far the fastest -- however the assignment asks for harmonic closeness
def classic_closeness(G: rx.PyGraph) -> np.ndarray:
    return np.fromiter(rx.closeness_centrality(G).values(), dtype=float)


def dijkstra_all_pairs_closeness(G: rx.PyGraph) -> np.ndarray:
    N = G.num_nodes()
    dist_dict = rx.all_pairs_dijkstra_path_lengths(G, edge_cost_fn=lambda _: 1)
    closeness = np.zeros(N)
    for u, dists in dist_dict.items():
        inv_dists = 1 / np.fromiter(dists.values(), dtype=float)
        closeness[u] = (1 / (N - 1)) * inv_dists.sum()
    return closeness


def nx_closeness(G: nx.Graph) -> np.ndarray:
    N = G.number_of_nodes()
    # nx.harmonic_centrality gives sum of reciprocals pero no normalization
    closeness = np.fromiter(nx.harmonic_centrality(G).values(), float) / (N - 1)
    return closeness


def early_stopping_closeness(
    G: rx.PyGraph, to_comp: float, batch_size: int = 500
) -> bool:

    N = G.num_nodes()
    all_nodes = set(G.node_indices())
    processed = set()
    degrees = {u: G.degree(u) for u in all_nodes}
    deg_sum_total = sum(degrees.values())
    deg_sum_done = 0
    closeness_i = np.zeros(N)

    parents = defaultdict(list)
    M = 0
    for u, d in degrees.items():
        if d == 1:
            parent = next(iter(G.neighbors(u)))
            parents[parent].append(u)

    for p, leafs in parents.items():
        parent_dists = rx.dijkstra_shortest_path_lengths(
            G, p, edge_cost_fn=lambda _: 1.0
        )
        closeness_i[p] = sum(1.0 / d for d in parent_dists.values() if d > 0) / (N - 1)
        all_nodes.discard(p)
        processed.add(p)
        M += 1
        deg_sum_done += degrees[p]

        for leaf in leafs:
            if leaf in processed:
                continue
            # the plus one is because the leaf is 1 hop away from parent and the parent_dists does not include it
            closeness_i[leaf] = (
                1 + sum(1.0 / (d + 1) for u, d in parent_dists.items() if u != leaf)
            ) / (N - 1)
            all_nodes.discard(leaf)
            processed.add(leaf)
            M += 1
            deg_sum_done += degrees[leaf]

    def eval_bounds():
        S = closeness_i.sum()

        deg_sum_rest = deg_sum_total - deg_sum_done
        Cmax = (S / N) + (N - M) / (2 * N) + (deg_sum_rest / (2 * N * (N - 1)))
        if Cmax < to_comp:
            print(f"cmax < to_comp: Processed {M} nodes")
            return "lt"
        Cmin = (S + deg_sum_rest / (N - 1)) / N
        if Cmin >= to_comp:
            print(f"cmin >= to_comp: Processed {M} nodes")
            return "gt"
        return None

    if M > batch_size:
        res = eval_bounds()
        if res is not None:
            return res == "gt"

    rest = sorted(all_nodes, key=lambda x: degrees[x])
    for M, u in enumerate(rest, start=M + 1):
        if M % batch_size == 0:
            res = eval_bounds()
            if res is not None:
                return res == "gt"
        dists = rx.dijkstra_shortest_path_lengths(G, u, edge_cost_fn=lambda _: 1.0)
        closeness_i[u] = sum(1.0 / d for d in dists.values()) / (N - 1)
        deg_sum_done += degrees[u]

    return eval_bounds() == "gt"


CLOSENESS_FUNCTIONS = {
    "classic": classic_closeness,
    "dijkstra_all_pairs": dijkstra_all_pairs_closeness,
    "nx": nx_closeness,
    "early_stopping": early_stopping_closeness,
}


def _validate_params(
    null_model: str, closeness_fn: str, Q: int = None, E: int = None
) -> tuple[str, str]:
    if not null_model in NULL_MODELS:
        raise ValueError(
            f"Unknown null model: {null_model} Use one of {list(NULL_MODELS.keys())}"
        )
    if not closeness_fn in CLOSENESS_FUNCTIONS:
        raise ValueError(
            f"Unknown closeness function: {closeness_fn} Use one of {list(CLOSENESS_FUNCTIONS.keys())}"
        )
    if not (null_model != "ES" or Q is not None):
        raise ValueError(
            "Q parameter must be provided for edge_switching (ES) null model"
        )
    if null_model == "ES" and Q < math.ceil(math.log(E)):
        print(
            bcolors.WARNING
            + f"Warning: Q={Q} is quite low for E={E} edges. Consider increasing Q to at least log(E)={math.ceil(math.log(E))} for better randomization."
            + bcolors.ENDC
        )
    if (null_model == "ER_NX" and closeness_fn != "nx") or (
        null_model != "ER_NX" and closeness_fn == "nx"
    ):
        print(
            bcolors.WARNING
            + "Warning: null_model and closeness_fn have been set to ER_NX and nx respectively, mixing RustworkX and NetworkX is not supported."
            + bcolors.ENDC
        )
        null_model, closeness_fn = "ER_NX", "nx"

    return null_model, closeness_fn


def simulate_closeness_significance(
    edges: set[tuple[str, str]],
    N: int,
    E: int,
    *,
    Q: int = None,
    T: int = 10000,
    null_model: str = "ER",
    closeness_fn: str = "classic",
    seed: int = None,
    benchmark: bool = False,
    early_stop_batch_size: int = 500,
) -> tuple[float, float, float]:

    null_model, closeness_fn = _validate_params(null_model, closeness_fn, Q, E)

    if null_model == "ER_NX":
        G = nx.Graph()
        G.add_edges_from(edges)
    else:
        G = rx.PyGraph(multigraph=False, node_count_hint=N, edge_count_hint=E)
        G.extend_from_edge_list(edges)

    if closeness_fn == "early_stopping":
        # early stopping uses harmonic closeness but since it doesnt fully evaluate closeness we compute it with the all pairs dijkstra variant
        orig_closeness = dijkstra_all_pairs_closeness(G)
    else:
        orig_closeness = CLOSENESS_FUNCTIONS[closeness_fn](G)

    avg_orig_closeness = orig_closeness.sum() / N

    base_params = {"N": N, "E": E}
    if null_model == "ES":
        base_params.update({"edges": list(edges), "Q": Q})

    base_seed = random.randint(0, 2**32 - 1) if seed is None else seed
    pbar_ctx = (
        tqdm_joblib(tqdm(total=T, desc="Simulations", leave=False, position=0))
        if benchmark
        else nullcontext()
    )

    with joblib.parallel_config(backend="loky", inner_max_num_threads=1), pbar_ctx:
        out = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(evaluate_one)(
                NULL_MODELS[null_model],
                CLOSENESS_FUNCTIONS[closeness_fn],
                {**base_params, "seed": base_seed + i},
                orig_avg_closeness=(
                    avg_orig_closeness if closeness_fn == "early_stopping" else None
                ),
                early_stop_batch_size=early_stop_batch_size,
            )
            for i in range(T)
        )

    if closeness_fn == "early_stopping":
        # in this case out is not a series of avg closeness but bools -> xNH >= avg_orig_closeness as the actual closeness is not computed
        f_xNH = sum(1 for x in out if x)
        return f_xNH / T, avg_orig_closeness, None, None

    f_xNH = sum(1 for x in out if x >= avg_orig_closeness)
    return f_xNH / T, avg_orig_closeness, np.mean(out), np.std(out)


def evaluate_one(
    null_model: callable,
    closeness_fn: callable,
    null_params: dict,
    orig_avg_closeness: float = None,
    early_stop_batch_size: int = 500,
) -> float:
    null_G = null_model(**null_params)

    # comparison mode -> full closeness is not computed only x_NH >= orig_avg_closeness by bounding
    if orig_avg_closeness is not None:
        is_greater = closeness_fn(
            null_G, orig_avg_closeness, batch_size=early_stop_batch_size
        )
        return is_greater

    closeness: np.ndarray = closeness_fn(null_G)
    avg_closeness = closeness.sum() / null_params["N"]
    return avg_closeness


if __name__ == "__main__":
    from src.graph_extraction import load_edges, get_network_summary
    from src.utils import LANG_DICT

    import time

    def benchmark_closeness_fns(closeness_fn: str):
        outs = []
        for lang in LANG_DICT.keys():

            edges = load_edges(lang=lang, int_optimize=True)
            N, E, _, _ = get_network_summary(lang)
            print(f"Processing {lang} with {N} nodes and {E} edges")

            outs.append(
                simulate_closeness_significance(
                    edges,
                    N,
                    E,
                    T=10,
                    null_model="ER",
                    closeness_fn=closeness_fn,
                    seed=42,
                )
            )
        return outs

    def benchmark(fn: callable):
        t0 = time.perf_counter()
        out = fn()
        return time.perf_counter() - t0, out

    for fn_name in CLOSENESS_FUNCTIONS.keys():
        duration, _ = benchmark(lambda: benchmark_closeness_fns(fn_name))
        print(f"{fn_name} benchmark duration: {duration:.4f} seconds")
