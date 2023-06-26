import time
from itertools import islice

import gurobipy as gb
import networkx as nx
import numpy as np


def get_k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_keys_edge(edge, paths):
    keys = []
    for key in paths:
        if (edge in paths[key]) or ((edge[1], edge[0]) in paths[key]):
            keys.append(key)
    return keys


def generate_random_matrix(num_nodes=10, episode_length=100, request_probabilities=None):
    if request_probabilities is None:
        request_probabilities = np.full(num_nodes, fill_value=1.0 / num_nodes)
    src = []
    dst = []
    for _ in range(episode_length):
        tmp = np.random.choice(num_nodes, p=request_probabilities)
        src.append(tmp)
        new_request_probabilities = np.copy(request_probabilities)
        new_request_probabilities[tmp] = 0
        new_request_probabilities = new_request_probabilities / np.sum(new_request_probabilities)
        tmp = np.random.choice(num_nodes, p=new_request_probabilities)
        dst.append(tmp)
    return np.array(src), np.array(dst)


def read_txt_file(file):
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


def get_topology(file_name, k_paths=3, num_wavs=80, request_probabilities=None, num_requests=1000, tm_path=None):
    topology = read_txt_file(file_name)

    topology.graph["node_indices"] = []
    topology.graph["num_wavs"] = num_wavs

    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx

    assert request_probabilities is None or len(request_probabilities) == topology.number_of_nodes()
    if request_probabilities is None:
        request_probabilities = np.full((topology.number_of_nodes()), fill_value=1.0 / topology.number_of_nodes())

    requests = {}

    if tm_path is None:
        src, dst = generate_random_matrix(
            num_nodes=topology.number_of_nodes(), episode_length=num_requests, request_probabilities=request_probabilities
        )
    else:
        tm = np.loadtxt(tm_path, dtype=np.int64, delimiter=",")
        src = tm[:, 0]
        dst = tm[:, 1]

    for i in range(num_requests):
        if src[i] < dst[i]:
            node_pair = (src[i], dst[i])
        else:
            node_pair = (dst[i], src[i])
        if node_pair in requests:
            requests[node_pair] += 1
        else:
            requests[node_pair] = 1
    topology.graph["requests"] = requests

    k_shortest_paths = {}
    k_shortest_paths_dict = {}
    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths)
                objs = []
                for path in paths:
                    objs.append(idp)
                    k_shortest_paths_dict[idp] = list(nx.utils.pairwise(path))
                    idp += 1
                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs

    topology.graph["ksp"] = k_shortest_paths
    topology.graph["ksp_dict"] = k_shortest_paths_dict
    topology.graph["k_paths"] = k_paths

    return topology


def RWA_model(topology: nx.Graph):
    model = gb.Model("RWA")
    x_cp = model.addVars(topology.graph["num_wavs"], len(topology.graph["ksp_dict"]), vtype=gb.GRB.BINARY, name="x")
    obj = gb.quicksum(
        x_cp[wav, p]
        for sd in topology.graph["requests"]
        for wav in range(topology.graph["num_wavs"])
        for p in topology.graph["ksp"][sd]
    )

    model.setObjective(obj, gb.GRB.MAXIMIZE)

    for edge in topology.edges():
        keys = get_keys_edge(edge, topology.graph["ksp_dict"])
        for wav in range(topology.graph["num_wavs"]):
            model.addConstr(gb.quicksum(x_cp[wav, p] for p in keys) <= 1)

    for sd in topology.graph["requests"]:
        model.addConstr(
            gb.quicksum(x_cp[wav, p] for wav in range(topology.graph["num_wavs"]) for p in topology.graph["ksp"][sd])
            <= topology.graph["requests"][sd]
        )
    return model


num_requests = 800
num_iter = 100
solutions = np.zeros((num_iter,))
variables = []
for i in range(num_iter):
    tm_path = f"./TM_RWA/gabriel/TM-{i+1}.txt"
    G = get_topology("./topologies/gabriel.txt", k_paths=3, num_wavs=80, num_requests=num_requests, tm_path=tm_path)
    model = RWA_model(G)
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", 15 * 60)
    start = time.time()
    model.optimize()
    print(f"Time: {time.time() - start}")
    solutions[i] = model.objVal
    print(f"Iteration: {i}")
solutions = (num_requests - solutions) / num_requests
print(f"Average blocking rate: {np.mean(solutions)}")
np.savetxt("./results/rwa-ilp/GEANT2.txt", solutions, delimiter=",")
