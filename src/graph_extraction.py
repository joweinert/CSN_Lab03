import os
import gzip

from src.utils import LANG_DICT


def parse_ud_conllu(
    data_folder: str = "./data/PUD",
    lang: str = "en",
    directed: bool = False,
    use_lemma: bool = True,
    rm_self_loops: bool = True,
    rm_punct: bool = True,
):
    def prep_line(line: str, sentence_id: int) -> bool:
        if not line.strip():
            sentence_id += 1
            return None, sentence_id
        if line.startswith("#"):
            return None, sentence_id
        words = line.split("\t")
        if (
            len(words) < 8
            or (rm_punct and (words[3] == "PUNCT" or words[7] == "punct"))
            or "-" in words[0]
            or "." in words[0]
            or not words[0].isdigit()
            or not words[6].isdigit()
        ):
            return None, sentence_id
        return words, sentence_id

    with open(data_folder + f"/{lang}_pud-ud-test.conllu", "r", encoding="utf-8") as f:

        lines = f.readlines()
        id_to_word_map = {}
        sentence_id = 0
        # mapping from (sentence_id, word_id) to word form or lemma
        for line in lines:
            words, sentence_id = prep_line(line, sentence_id)
            if words is None:
                continue
            id_to_word_map[(sentence_id, int(words[0]))] = (
                words[2] if use_lemma else words[1]
            )
        # second pass to extract edges using the maping
        edges = set()
        sentence_id = 0
        for line in lines:
            words, sentence_id = prep_line(line, sentence_id)
            if words is None:
                continue

            parent = int(words[6])
            if parent != 0:  # root!
                child_id = int(words[0])
                parent_word = id_to_word_map[(sentence_id, parent)]
                child_word = id_to_word_map[(sentence_id, child_id)]

                if rm_self_loops and parent_word == child_word:
                    continue
                if directed:
                    edges.add((parent_word, child_word))
                else:
                    # sorting the tuple to have undirected graph encoded in parsing already
                    edges.add(tuple(sorted((parent_word, child_word))))

        nodes = set()
        for parent, child in edges:
            nodes.add(parent)
            nodes.add(child)

        return (nodes, edges)


def save_adjacency_list(
    nodes: set[str],
    edges: set[tuple[str, str]],
    lang: str = "en",
    *,
    compresslevel: int = 6,
    directed: bool = False,
    output_folder: str = "./data/dependency_networks/adjacency_lists",
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    path = output_folder + f"/{LANG_DICT[lang]}_dependency_network.txt.gz"

    adjacency = {n: set() for n in nodes}
    for u, v in edges:
        adjacency[u].add(v)
        if not directed:
            adjacency[v].add(u)

    N = len(nodes)
    E = len(edges)
    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
        compresslevel=compresslevel,
    ) as f:
        f.write(f"{N}\t{E}\n")
        for n in sorted(adjacency.keys()):
            neighbors = "\t".join(sorted(adjacency[n]))
            f.write(f"{n}\t{neighbors}\n")


def save_edge_list(
    nodes: set[str],
    edges: set[tuple[str, str]],
    lang: str = "en",
    *,
    int_optimize: bool = False,
    compresslevel: int = 6,
    output_folder: str = "./data/dependency_networks/edge_lists",
):

    if int_optimize:
        output_folder += "_int_optimize"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename_base = f"/{LANG_DICT[lang]}_dependency_network"
    file_extension = ".txt.gz"
    N = len(nodes)
    E = len(edges)
    if int_optimize:
        node_to_id = {n: i for i, n in enumerate(sorted(nodes))}
        edges = {(node_to_id[u], node_to_id[v]) for u, v in edges}
        filename = filename_base + "_edges"
    else:
        filename = filename_base

    with gzip.open(
        output_folder + filename + file_extension,
        mode="wt",
        encoding="utf-8",
        compresslevel=compresslevel,
    ) as f:
        f.write(f"{N}\t{E}\n")
        for u, v in sorted(edges):
            f.write(f"{u}\t{v}\n")

    if not int_optimize:
        return

    filename = filename_base + "_node_to_id"
    with gzip.open(
        output_folder + filename + file_extension,
        mode="wt",
        encoding="utf-8",
        compresslevel=compresslevel,
    ) as f:
        for n, i in sorted(node_to_id.items(), key=lambda x: x[1]):
            f.write(f"{i}\t{n}\n")


def load_edges(
    lang: str = "en",
    *,
    int_optimize: bool = False,
    input_folder: str = "./data/dependency_networks/edge_lists",
) -> set[tuple[str, str]]:
    filename = f"/{LANG_DICT[lang]}_dependency_network"
    file_extension = ".txt.gz"
    if int_optimize:
        input_folder += "_int_optimize"
        filename += "_edges"
    path = input_folder + filename + file_extension

    edges = set()
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        first_line = f.readline()
        N, E = map(int, first_line.strip().split("\t"))
        for line in f:
            u, v = line.strip().split("\t")
            if int_optimize:
                u, v = int(u), int(v)
            edges.add((u, v))

    return edges


def get_network_summary(
    lang, input_folder: str = "./data/dependency_networks/edge_lists"
):
    filename = f"/{LANG_DICT[lang]}_dependency_network"
    file_extension = ".txt.gz"
    with gzip.open(
        input_folder + filename + file_extension, mode="rt", encoding="utf-8"
    ) as f:
        first_line = f.readline()
        N, E = map(int, first_line.strip().split("\t"))

    k = 2 * E / N  # average degree
    delta = 2 * E / (N * (N - 1))  # density
    return (
        N,
        E,
        k,
        delta,
    )


if __name__ == "__main__":
    for lang in LANG_DICT.keys():
        print(f"Processing language: {lang}")
        nodes, edges = parse_ud_conllu(
            lang=lang, directed=False, use_lemma=True, rm_self_loops=True, rm_punct=True
        )
        save_adjacency_list(nodes, edges, lang=lang, directed=False)
        save_edge_list(nodes, edges, lang=lang, int_optimize=True)
