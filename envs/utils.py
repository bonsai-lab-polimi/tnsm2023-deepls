import time
from dataclasses import dataclass
from itertools import islice
from typing import Optional, Union

import gym
import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.algorithms.shortest_paths.weighted import _bellman_ford, _dijkstra, _weight_function
from networkx.classes.function import path_weight
from tqdm import tqdm


@dataclass
class Path:
    node_list: list[int]
    length: float


@dataclass
class Request:
    request_id: int
    arrival_time: float
    holding_time: float
    src: int
    dst: int
    bitrate: Optional[int] = None
    n_slots: Optional[int] = None
    route: Optional[Path] = None
    first_slot: Optional[int] = None


def read_txt_file(file: str) -> nx.Graph:
    graph = nx.Graph()
    id_link = 0
    with open(file, "r") as lines:
        # gets only lines that do not start with the # character
        nodes_lines = [value for value in lines if not value.startswith("#")]
        for idx, line in enumerate(nodes_lines):
            if idx == 0:
                continue
            elif idx == 1:
                continue
            elif len(line) > 1:
                info = line.replace("\n", "").split(" ")
                graph.add_edge(info[0], info[1], id=id_link, index=id_link, weight=1, length=int(info[2]))
                id_link += 1
    graph = nx.convert_node_labels_to_integers(graph)
    return graph


def read_topology(filepath: str, k_paths: int = 5) -> nx.Graph:
    topology = read_txt_file(filepath)

    link_id_dict = {}
    for idx, (i, j) in enumerate(topology.edges()):
        topology[i][j]["id"] = idx
        link_id_dict[idx] = (i, j)
    topology.graph["link_id_dict"] = link_id_dict

    path_edge_dict = {}
    k_shortest_paths = {}
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = list(islice(nx.shortest_simple_paths(topology, n1, n2, weight="length"), k_paths))
                lengths = [path_weight(topology, path, weight="length") for path in paths]
                ksp = []
                path_edges = set()
                for path, length in zip(paths, lengths):
                    for edge in zip(path, path[1:]):
                        path_edges.add(edge)
                    ksp.append(Path(path, length))
                k_shortest_paths[n1, n2] = ksp
                k_shortest_paths[n2, n1] = ksp
                path_edge_dict[n1, n2] = path_edges
                path_edge_dict[n2, n1] = path_edges
        topology.graph["ksp"] = k_shortest_paths
        topology.graph["default_ksp"] = k_shortest_paths
        topology.graph["default_path_edge_dict"] = path_edge_dict
        topology.graph["path_edge_dict"] = path_edge_dict
        topology.graph["k_paths"] = k_paths
    return topology


def evaluate_random_policy(env: gym.Env, n_episodes: int = 10) -> npt.NDArray:
    tic = time.time()
    episode_returns = np.zeros((n_episodes,))
    for i in tqdm(range(n_episodes)):
        env.reset()
        done = False
        ret = 0
        while not done:
            _, reward, done, _ = env.step(env.action_space.sample())
            ret += reward
        episode_returns[i] = ret
    toc = time.time()
    print(f"Time elapsed: {toc - tic}")
    print(f"Average return: {np.mean(episode_returns)}")
    return np.array(episode_returns)


def metropolis_acceptance(delta: float, temperature: float) -> float:
    return np.exp(-delta / temperature)


def get_k_shortest_paths(G: Union[nx.Graph, nx.DiGraph], source: int, target: int, k: int, weight: str = None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def johnson_all_sp(G: Union[nx.Graph, nx.DiGraph], weight="weight"):
    # Code largely borrowed from nx.johnson
    # Outputs dictionary of predecessors
    if not nx.is_weighted(G, weight=weight):
        raise nx.NetworkXError("Graph is not weighted.")

    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}
    weight = _weight_function(G, weight)

    # Calculate distance of shortest paths
    dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

    # Update the weight function to take into account the Bellman-Ford
    # relaxation distances
    def new_weight(u, v, d):
        return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

    def dist_path(v):
        pred = {v: [v]}
        _dijkstra(G, v, new_weight, pred=pred)
        return pred

    return {v: dist_path(v) for v in G}
