import numpy as np
import networkx as nx
from collections import Counter, deque
from itertools import islice


def recolor_bfs(colors: dict, graph: nx.Graph) -> dict:
    """
    Use the information that 0 and 1 color will be close for us, to make image more distinguishable
    Parameters
    ----------
    colors: dict, old coloring
    graph: nx.Graph, graph of object connections

    Returns
    -------
    dict, new coloring, that uses that 0 and 1 is close colors
    """
    if not colors or not len(graph.nodes):
        return {}
    amount_of_colors = len(set(colors.values()))
    colors_range = deque(range(amount_of_colors))
    colors_freqs = Counter(colors.values())

    most_common_color = colors_freqs.most_common(1)[0][0]
    result = {most_common_color: colors_range.pop()}

    # find starting point - any node of most_common_color
    current_point = None
    for v in graph.nodes:
        if colors[v] == most_common_color:
            current_point = v
            break
    assert current_point is not None

    stack = deque()
    while len(result) < amount_of_colors:
        neighs = graph.neighbors(current_point)
        neighs = [v for v in neighs if colors[v] not in result]

        neighs_colors = {colors[v] for v in neighs}
        if neighs_colors:
            step = len(colors_range) // len(neighs_colors)
            colors_to_use = deque(islice(colors_range, None, None, step))
            assert len(colors_to_use) >= len(neighs_colors)  # maybe even ==
            for v in neighs:
                if colors[v] not in result:
                    result[colors[v]] = colors_to_use.pop()
                    colors_range.remove(result[colors[v]])

            colors_range.reverse()

        stack.extend(neighs)
        current_point = stack.popleft()

    return {k: result[v] for k, v in colors.items()}


def recolor_by_frequency(colors: dict) -> dict:
    """
    Use the information that 0 and 1 color will be close for us and info about the frequencies of similar objects
    to change the colors to new ones
    Args:
        colors: dict, old coloring

    Returns:
        dict, new coloring, that uses that 0 and 1 is close colors
    """
    replace_dict = {val: ind for ind, val in
                    enumerate(sorted(set(colors.values()), key=lambda x: list(colors.values()).count(colors[x])))}
    result_dict = {}
    for key in colors:
        result_dict[key] = replace_dict[colors[key]]
    return result_dict
