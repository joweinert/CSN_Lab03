from collections import defaultdict
import math
import rustworkx as rx
import networkx as nx
import joblib
import random
import numpy as np

from src.graph_extraction import load_edges, get_network_summary
from src.utils import bcolors, LANG_DICT, fmt_simulation_results
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
    return np.fromiter(
        rx.closeness_centrality(G).values(), dtype=float, count=G.num_nodes()
    )


def dijkstra_all_pairs_closeness(G: rx.PyGraph) -> np.ndarray:
    N = G.num_nodes()
    dist_dict = rx.all_pairs_dijkstra_path_lengths(G, edge_cost_fn=lambda _: 1)
    closeness = np.zeros(N)
    for u, dists in dist_dict.items():
        inv_dists = 1 / np.fromiter(dists.values(), dtype=float, count=len(dists))
        closeness[u] = (1 / (N - 1)) * inv_dists.sum()
        # s = 0.0
        # for d in dists.values():
        #     s += 1.0 / d
        # closeness[u] = s / (N - 1)
    return closeness


def nx_closeness(G: nx.Graph) -> np.ndarray:
    N = G.number_of_nodes()
    # nx.harmonic_centrality gives sum of reciprocals pero no normalization
    closeness = np.fromiter(nx.harmonic_centrality(G).values(), float, count=N) / (
        N - 1
    )
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

    res = np.zeros(T)
    for i in range(T):
        null_G = NULL_MODELS[null_model](**{**base_params, "seed": base_seed + i})

        # comparison mode -> full closeness is not computed only x_NH >= orig_avg_closeness by bounding
        if closeness_fn == "early_stopping":
            res[i] = closeness_fn(
                null_G, avg_orig_closeness, batch_size=early_stop_batch_size
            )
            continue

        res[i] = CLOSENESS_FUNCTIONS[closeness_fn](null_G).sum() / N

    if closeness_fn == "early_stopping":
        # in this case out is not a series of avg closeness but bools -> xNH >= avg_orig_closeness as the actual closeness is not computed
        f_xNH = sum(1 for x in res if x)
        return f_xNH / T, avg_orig_closeness, None, None

    f_xNH = sum(1 for x in res if x >= avg_orig_closeness)
    return f_xNH / T, avg_orig_closeness, np.mean(res), np.std(res)


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


def run_one_language(lang, null_models_to_run, T, Q, closeness_fn, seed):
    results = defaultdict(str)
    edges = load_edges(lang=lang, int_optimize=True)
    N, E, *_ = get_network_summary(lang)
    for nm in null_models_to_run:
        results[nm] = simulate_closeness_significance(
            edges,
            N,
            E,
            T=T,
            Q=Q,
            null_model=nm,
            closeness_fn=closeness_fn,
            seed=seed,
        )
    print(fmt_simulation_results(lang, results))


if __name__ == "__main__":
    import argparse
    import time
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--Q", type=int, default=20)
    parser.add_argument(
        "--null_model", type=str, choices=["ER", "ES", "BOTH"], default="BOTH"
    )
    parser.add_argument(
        "--closeness_fn",
        type=str,
        choices=list(CLOSENESS_FUNCTIONS.keys()),
        default="dijkstra_all_pairs",
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    header = f"{'Language':<15}{'nullmodel':<15}{'pvalue':<15}{'avg_orig_closeness':<25}{'avg_null_closeness':<25}{'std_null_closeness':<25}\n{'_'*120}"
    print(header)

    null_models_to_run = (
        ["ER", "ES"] if args.null_model == "BOTH" else [args.null_model]
    )

    num_cpus = joblib.cpu_count() or 1
    n_jobs_lang = max(1, int(np.ceil(np.sqrt(num_cpus))))
    rayon_threads = max(1, int(np.floor(num_cpus / n_jobs_lang)))
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)

    langs = list(LANG_DICT.keys())

    t0 = time.perf_counter()
    with joblib.parallel_config(n_jobs=n_jobs_lang):
        joblib.Parallel()(
            joblib.delayed(run_one_language)(
                lang, null_models_to_run, args.T, args.Q, args.closeness_fn, args.seed
            )
            for lang in langs
        )
    t1 = time.perf_counter()

    print(f"\nTotal duration: {t1 - t0:.4f} seconds")
