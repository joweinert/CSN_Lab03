import rustworkx as rx
import joblib
from contextlib import nullcontext
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import random
import numpy as np

from src.null_model import erdos_renyi_null_model, edge_switching_null_model

NULL_MODELS = {
    "ER": erdos_renyi_null_model,
    "ES": edge_switching_null_model,
}


def simulate_closeness_significance(
    edges: set[tuple[str, str]],
    N: int,
    E: int,
    *,
    Q: int = None,
    T: int = 10000,
    null_model: str = "ER",
    seed: int = None,
    benchmark: bool = False,
) -> tuple[float, float, float]:
    assert null_model in NULL_MODELS, f"Unknown null model: {null_model}"
    assert (
        null_model != "ES" or Q is not None
    ), "Q parameter must be provided for edge_switching (ES) null model"

    G = rx.PyGraph(multigraph=False)
    G.extend_from_edge_list(edges)

    orig_closeness = rx.closeness_centrality(G)
    avg_orig_closeness = sum(orig_closeness.values()) / N

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
        avgs = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(get_one_avg_closeness)(
                NULL_MODELS[null_model],
                {**base_params, "seed": base_seed + i},
            )
            for i in range(T)
        )

    f_xNH = sum(1 for x in avgs if x >= avg_orig_closeness)
    return f_xNH / T, avg_orig_closeness, np.mean(avgs), np.std(avgs)


def get_one_avg_closeness(null_model: callable, null_params: dict) -> float:
    null_G = null_model(**null_params)
    closeness = rx.closeness_centrality(null_G)
    assert (
        len(closeness) == null_params["N"]
    ), "Closeness centrality size does not match N"
    avg_closeness = sum(closeness.values()) / null_params["N"]
    return avg_closeness
