import copy
import functools
import os
from itertools import islice
from typing import Optional

import gym.spaces
import networkx as nx
import numpy as np
from networkx import edge_betweenness_centrality
from networkx.classes.function import path_weight

from .utils import Path, read_topology


class RWAEnv(gym.Env):
    def __init__(
        self,
        network: str,
        episode_length: int = 10,
        n_requests: int = 100,
        n_slots: int = 10,
        request_proba: Optional[np.ndarray] = None,
        seed: int = 42,
        eval: bool = False,
    ):
        self.network = network
        self.topology = read_topology(f"./topologies/{self.network}.txt", k_paths=3)
        self.episode_length = episode_length
        self.n_requests = n_requests
        self.n_slots = n_slots
        self.request_proba = request_proba
        if self.request_proba is None:
            self.request_proba = np.full(self.topology.number_of_nodes(), 1 / self.topology.number_of_nodes())
        self.eval = eval
        self.dataset_dir = os.path.join(".", "TM_RWA", self.network)
        self.num_sample = 0

        # self.action_space = gym.spaces.MultiBinary(self.topology.number_of_edges())
        self.action_space = gym.spaces.Discrete(self.topology.number_of_edges())
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.topology.number_of_edges(), 3), dtype=np.float32)

        # Set default weights
        self.default_weights = 1.0

        self.seed(seed=seed)

        self.episode_counter = 0

        self.reset()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        # Set new edge weights
        i, j = self.topology.graph["link_id_dict"][int(action)]
        self.weights[action] += 1
        self.topology[i][j]["length"] += 1
        # Recompute shortest paths
        self._update_ksp(action)
        # Route using ksp
        self.blocking_rate, available_slots = self._ksp_ff()

        reward = self.reward(available_slots)

        info = {
            "blocking_rate": self.blocking_rate,
            "ksp_blocking_rate": self.ksp_blocking,
            "best_blocking": self.best_blocking,
        }

        self.iter_count += 1

        self.temperature = self.temperature * 0.99

        return self.observation(), reward, (self.iter_count >= self.episode_length), info

    def observation(self) -> np.ndarray:
        centrality = np.array(list(edge_betweenness_centrality(self.topology, normalized=True).values()))
        normalized_capacity = np.sum(self.available_slots, axis=-1) / self.n_slots
        normalized_weights = self.weights / np.max(self.weights)
        return np.concatenate(
            [normalized_capacity.reshape(-1, 1), normalized_weights.reshape(-1, 1), centrality.reshape(-1, 1)],
            axis=-1,
            dtype=np.float32,
        )

    def reward(self, available_slots: np.ndarray) -> float:
        if self.blocking_rate < self.best_blocking:
            reward = self.current_blocking - self.blocking_rate
            self.best_blocking = self.blocking_rate
            self.current_blocking = self.blocking_rate
            self.best_weights = self.weights
            self.available_slots = available_slots
        else:
            # if self.rng.random() < self._metropolis_acceptance():
            reward = self.current_blocking - self.blocking_rate
            self.current_blocking = self.blocking_rate
            self.available_slots = available_slots
            # else:
            #     for i, edge in enumerate(self.topology.edges()):
            #         self.topology[edge[0]][edge[1]]['length'] = self.best_weights[i]
            #     reward = self.current_blocking - self.blocking_rate
        return reward

    def reset(self) -> np.ndarray:
        self.episode_requests_processed = 0
        self.episode_requests_accepted = 0

        self.available_slots = np.full(shape=(self.topology.number_of_edges(), self.n_slots), fill_value=1, dtype=np.uint8)

        self.iter_count = 0

        if not self.eval:
            self.demands = self._generate_demands()
        else:
            self.demands = self._load_traffic_matrix()
            self.num_sample += 1

        nx.set_edge_attributes(self.topology, self.default_weights, "length")

        self.topology.graph["ksp"] = copy.deepcopy(self.topology.graph["default_ksp"])
        self.topology.graph["path_edge_dict"] = copy.deepcopy(self.topology.graph["default_path_edge_dict"])
        self.ksp_blocking, self.available_slots = self._ksp_ff()
        self.current_blocking = self.ksp_blocking

        self.weights = np.ones((self.topology.number_of_edges(),))
        self.best_blocking = self.ksp_blocking
        self.best_weights = np.ones(self.topology.number_of_edges())

        self.temperature = 80

        return self.observation()

    def seed(self, seed: int) -> None:
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.rng = np.random.default_rng(seed)

    def _ksp_ff(self):
        # Path are already sorted by length
        available_slots = np.full((self.topology.number_of_edges(), self.n_slots), fill_value=1, dtype=np.uint8)
        requests_accepted = 0
        for source, destination in self.demands:
            decision = (self.topology.graph["k_paths"], self.n_slots)
            for idp, path in enumerate(self.topology.graph["ksp"][source, destination]):
                path_free_slots = np.where(self._get_path_available_wavelengths(path, available_slots))[0]
                if path_free_slots.size != 0:
                    decision = (idp, path_free_slots[0])
                    break
            if decision[0] < self.topology.graph["k_paths"]:
                path = self.topology.graph["ksp"][source, destination][decision[0]]
                for i in range(len(path.node_list) - 1):
                    if available_slots[self.topology[path.node_list[i]][path.node_list[i + 1]]["id"], decision[1]] < 1:
                        raise ValueError("The selected link does not have enough capacity!")
                    available_slots[self.topology[path.node_list[i]][path.node_list[i + 1]]["id"], decision[1]] = 0
                requests_accepted += 1
        return (self.n_requests - requests_accepted) / self.n_requests, available_slots

    def _generate_demands(self) -> list[tuple[int, int]]:
        demands = []
        for _ in range(self.n_requests):
            src, dst = self._get_src_dst()
            # sp_length = self.topology.graph['ksp'][src, dst][0].length
            demands.append([src, dst])
        # dtype = [('src', int), ('dst', int), ('sp_length', float)]
        # demands = np.sort(np.array(demands, dtype=dtype), order='sp_length')
        self.episode_counter += 1
        return demands

    def _load_traffic_matrix(self) -> list[tuple[int, int]]:
        tm_file = os.path.join(self.dataset_dir, "TM-" + str(self.num_sample) + ".txt")
        demands = []
        with open(tm_file) as fd:
            for line in fd:
                camps = line.split(",")
                demands.append((int(camps[0]), int(camps[1])))
        return demands

    def _is_path_free(self, path: Path, available_slots: np.ndarray, slot: int):
        if slot >= self.n_slots:
            return False
        for i in range(len(path.node_list) - 1):
            if available_slots[self.topology[path.node_list[i]][path.node_list[i + 1]]["id"], slot] == 0:
                return False
        return True

    def _update_ksp(self, action: int, weight="length") -> None:
        for idn1, n1 in enumerate(self.topology.nodes()):
            for idn2, n2 in enumerate(self.topology.nodes()):
                if idn1 < idn2:
                    # if self.topology.graph["link_id_dict"][action] in self.topology.graph["path_edge_dict"][idn1, idn2]:
                    paths = list(
                        islice(nx.shortest_simple_paths(self.topology, n1, n2, weight=weight), self.topology.graph["k_paths"])
                    )
                    lengths = [path_weight(self.topology, path, weight=weight) for path in paths]
                    objs = []
                    path_edges = set()
                    for path, length in zip(paths, lengths):
                        for edge in zip(path, path[1:]):
                            path_edges.add(edge)
                        objs.append(Path(path, length))
                    self.topology.graph["ksp"][n1, n2] = objs
                    self.topology.graph["ksp"][n2, n1] = objs
                    self.topology.graph["path_edge_dict"][idn1, idn2] = path_edges
                    self.topology.graph["path_edge_dict"][idn1, idn2] = path_edges

    def _get_src_dst(self) -> tuple[int, int]:
        src, dst = self.rng.choice(a=self.topology.number_of_nodes(), size=(2,), replace=False, p=self.request_proba)
        return src, dst

    def _get_path_available_wavelengths(self, path: Path, available_slots: np.ndarray):
        available_wavelengths = functools.reduce(
            np.multiply,
            available_slots[
                [self.topology[path.node_list[i]][path.node_list[i + 1]]["id"] for i in range(len(path.node_list) - 1)], :
            ],
        )
        return available_wavelengths

    def _metropolis_acceptance(self) -> float:
        return np.exp(-(self.blocking_rate - self.current_blocking) / self.temperature)
