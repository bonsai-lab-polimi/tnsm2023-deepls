import os
from typing import Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB


def generate_dummy_graph():
    topology = nx.Graph()
    topology.add_edge(0, 1)
    topology.add_edge(1, 3)
    topology.add_edge(0, 2)
    topology.add_edge(2, 3)
    nx.set_edge_attributes(topology, 1, "capacity")
    topology = nx.to_directed(topology)
    link_id_dict = {}
    idx = 0
    for i, j in topology.edges():
        topology[i][j]["id"] = idx
        link_id_dict[idx] = (i, j)
        idx += 1
    traffic_demand = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    return topology, link_id_dict, traffic_demand


def load_topology(base_dir: str, dataset_dir: str, env_type: str) -> Tuple[nx.DiGraph, dict]:
    try:
        nx_file = os.path.join(base_dir, env_type, "graph_attr.txt")
        topology = nx.DiGraph(nx.read_gml(nx_file, destringizer=int))
    except:
        topology = nx.DiGraph()
        capacity_file = os.path.join(dataset_dir, "capacities", "graph.txt")
        with open(capacity_file) as fd:
            for line in fd:
                if "Link_" in line:
                    camps = line.split(" ")
                    topology.add_edge(int(camps[1]), int(camps[2]))
                    topology[int(camps[1])][int(camps[2])]["bandwidth"] = int(camps[4])
    link_id_dict = {}
    idx = 0
    for i, j in topology.edges():
        topology[i][j]["id"] = idx
        link_id_dict[idx] = (i, j)
        idx += 1
    return topology, link_id_dict


def set_capacities(G: nx.DiGraph, traffic_profile: str, dataset_dir: str, num_sample: int) -> nx.DiGraph:
    if traffic_profile == "gravity_full":
        capacity_file = os.path.join(dataset_dir, "capacities", "graph-TM-" + str(num_sample) + ".txt")
    else:
        capacity_file = os.path.join(dataset_dir, "capacities", "graph.txt")
    with open(capacity_file) as fd:
        for line in fd:
            if "Link_" in line:
                camps = line.split(" ")
                G[int(camps[1])][int(camps[2])]["capacity"] = int(camps[4])
    return G


def load_traffic_matrix(dataset_dir: str, num_sample: int) -> Tuple[np.ndarray, dict]:
    tm_file = os.path.join(dataset_dir, "TM", "TM-" + str(num_sample))
    traffic_demand = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    traffic_demand_dict = {}
    with open(tm_file) as fd:
        fd.readline()
        fd.readline()
        for line in fd:
            camps = line.split(" ")
            traffic_demand[int(camps[1]), int(camps[2])] = float(camps[3])
            traffic_demand_dict[str((int(camps[1]), int(camps[2])))] = float(camps[3])
    return traffic_demand, traffic_demand_dict


if __name__ == "__main__":
    base_dir = "./MARL-GNN-TE/datasets"
    env_type = "GEANT2"
    traffic_profile = "uniform"
    dataset_dir = os.path.join(base_dir, env_type, traffic_profile)
    optimal_min_max_loads = []
    for num_sample in range(99, 200):
        G, link_id_dict = load_topology(base_dir, dataset_dir, env_type)
        G = set_capacities(G, traffic_profile, dataset_dir, num_sample)
        traffic_demand, traffic_demand_dict = load_traffic_matrix(dataset_dir, num_sample)
        destinations = list(G.nodes())

        model = gp.Model("multicommodity")
        flow = model.addVars(destinations, G.edges(), name="flow")
        model.update()
        max_load = model.addVar(name="max_load")
        model.addConstrs((flow.sum("*", u, v) <= G[u][v]["capacity"] for u, v in G.edges()), "cap")
        model.addConstrs(
            (
                flow.sum(t, s, "*") - flow.sum(t, "*", s) == traffic_demand[s, t]
                for s in G.nodes()
                for t in G.nodes()
                if s != t
            ),
            "node",
        )
        model.addConstrs((max_load >= (flow.sum("*", u, v) / G[u][v]["capacity"]) for u, v in G.edges()), "set_max_load")

        model.setObjective(max_load, GRB.MINIMIZE)
        model.optimize()
        optimal_min_max_loads.append(model.ObjVal)
        # if model.Status == GRB.OPTIMAL:
        #     solution = model.getAttr('X', flow)
        #     print('\nOptimal flows:')
        #     for t in G.nodes():
        #         print(f"Destination {t}")
        #         for u, v in G.edges():
        #             print('%s -> %s: %g' % (u, v, solution[t, u, v]))
    np.savetxt(f"./results/lp_results/{env_type}-{traffic_profile}.txt", np.array(optimal_min_max_loads), delimiter=",")
