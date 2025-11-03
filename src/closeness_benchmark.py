import time

from src.monte_carlo import simulate_closeness_significance, CLOSENESS_FUNCTIONS
from src.graph_extraction import load_edges, get_network_summary
from src.utils import LANG_DICT


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
                T=1,
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


if __name__ == "__main__":
    for fn_name in CLOSENESS_FUNCTIONS.keys():
        duration, _ = benchmark(lambda: benchmark_closeness_fns(fn_name))
        print(f"{fn_name} benchmark duration: {duration:.4f} seconds")
