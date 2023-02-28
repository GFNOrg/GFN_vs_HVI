import numpy as np
import networkx as nx

from collections import defaultdict, deque

from dag_gflownet.utils.graph import get_children
from dag_gflownet.utils.gflownet import get_gflownet_cache


def push_source_flow_to_terminal_states(gfn_state_graph, source_state_graph):
    """Compute a hashable key for a graph.

    This function traverses the GFlowNet state-action space graph (DAG) in a 
    topologically sorted order and "pushes" the log_flow from each node to
    its children according to the log_prob_action specified on the edges.
    The topological sort ensures that all the flow has "arrived" at a node
    before "moving" its flow to its children.

    Parameters
    ----------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space where each node represents one GFlowNet
        state and each edge represents one GFlowNet action.

    source_state_graph: nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space but now each node has an attribute
        named log_flow, which is -np.inf for non-terminal states and
        the marginal log probability for the terminal states.
    """
    # Initialize log_flow to be -np.inf (flow = 0) for all nodes
    nx.set_node_attributes(gfn_state_graph, -np.inf, "log_flow")

    # Except initialize log_flow to be 0 (flow = 1) for source node
    source_node_key = frozenset(source_state_graph.edges)
    nx.set_node_attributes(gfn_state_graph, {source_node_key: 0}, "log_flow")

    # Push flow through sorted graph
    for state in nx.topological_sort(gfn_state_graph):
        current_node = gfn_state_graph.nodes[state]
        log_flow_incoming = current_node["log_flow"]  # log_flow is log probability of reaching this node starting from source node

        # Compute terminal_log_flow
        stop_action_log_flow = current_node["stop_action_log_flow"]  # probability of taking stop action from this node

        # terminal prob = incoming probability * p(stop action at this node)
        current_node["terminal_log_flow"] = log_flow_incoming + stop_action_log_flow

        # Push flow along edges to children
        edges = gfn_state_graph.edges(state, data=True)
        for _, child, edge_attr in edges:
            log_prob_action = edge_attr["log_prob_action"]
            existing_log_flow_child = gfn_state_graph.nodes[child]["log_flow"]
            updated_log_flow_child = np.logaddexp(
                existing_log_flow_child, log_flow_incoming + log_prob_action
            )
            nx.set_node_attributes(gfn_state_graph, {child: updated_log_flow_child}, "log_flow")

    return gfn_state_graph


def construct_state_dag_with_bfs(gflownet_cache, nodelist, source_graph=None):
    """Constructs the state-action space of the GFlowNet.

    This function performs Breadth-First Search on the GFlowNet state-action space
    starting from the source state, in order to construct a networkx.DiGraph object
    where each node is a GFlowNet state and each edge is labeled with the action
    and the log probability of taking that action. Each node is also labeled with
    the stop_action_log_flow which contains the probability of terminating at that state.

    Parameters
    ----------
    gflownet_cache :

    nodelist :

    source_graph : nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space.

    source_graph : nx.DiGraph instance
        The graph representing the source state.
    """
    gfn_state_graph = nx.DiGraph()
    is_state_queued = defaultdict(bool)
    states_to_visit = deque()

    if source_graph is None:
        source_graph = nx.DiGraph()
        source_graph.add_nodes_from(nodelist)
    source_graph_key = frozenset(source_graph.edges)

    gfn_state_graph.add_node(source_graph_key, graph=source_graph)
    states_to_visit.append(source_graph)
    is_state_queued[source_graph_key] = True
    while len(states_to_visit) > 0:
        current_graph = states_to_visit.popleft()
        current_graph_key = frozenset(current_graph.edges)
        children = get_children(current_graph, gflownet_cache, nodelist)
        for child_graph, action, log_prob in children:
            if action is None:  # stop action
                # Encode the stop action as a node attribute
                gfn_state_graph.nodes[current_graph_key]['stop_action_log_flow'] = log_prob
            else:
                child_graph_key = frozenset(child_graph.edges)
                if child_graph_key not in gfn_state_graph:
                    gfn_state_graph.add_node(child_graph_key, graph=child_graph)
                gfn_state_graph.add_edge(
                    current_graph_key,
                    child_graph_key,
                    action=action,
                    log_prob_action=log_prob
                )
                already_visited = is_state_queued[child_graph_key]
                if not already_visited:
                    states_to_visit.append(child_graph)
                    is_state_queued[child_graph_key] = True

    return gfn_state_graph, source_graph


def posterior_exact(gflownet, params, nodelist, batch_size=256):
    gfn_cache = get_gflownet_cache(gflownet, params, nodelist, batch_size)
    gfn_state_graph, source_state_graph = construct_state_dag_with_bfs(
        gfn_cache, nodelist)
    gfn_state_graph = push_source_flow_to_terminal_states(
        gfn_state_graph, source_state_graph)
    return gfn_state_graph
