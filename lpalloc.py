"""Bid-price / fluid LP allocation of stock to buyers over the remaining rounds (stdlib only).

Gallego & van Ryzin (1994) fluid bound, Besbes & Zeevi (2012) blind NRM, bandits-with-knapsacks LP: with limited
stock and limited time, solve the deterministic problem for the whole horizon and act on it now, re-solving every
round with the latest estimates.

    max  sum_{lot, buyer} x[l,b] * value[l,b]          (value = expected THB uplift of one bag over salvage)
    s.t. sum_b x[l,b] <= stock[l]                      (bags of each lot)
         sum_l x[l,b] <= demand[b]                     (bags this buyer can still take from us, rounds left)
         sum_{l,b} x[l,b] <= total_cap                 (round capacity x rounds left)
         x >= 0, only eligible (lot passes the buyer's spec)

It is a transportation problem, so min-cost flow solves it exactly in integers. Bags not placed go to salvage.
"""
from __future__ import annotations


def assign(stock: dict[str, int], demand: dict[str, int], value: dict[tuple[str, str], float],
           total_cap: int | None = None) -> dict[str, dict[str, int]]:
    """Returns {lot: {buyer: bags}} (only positive entries). value only for eligible pairs; <= 0 pairs are ignored."""
    lots, buyers = list(stock), list(demand)
    S, CAP, T = "S", "CAP", "T"
    nodes = [S, CAP, T] + [("L", l) for l in lots] + [("B", b) for b in buyers]
    idx = {n: i for i, n in enumerate(nodes)}
    graph: list[list[int]] = [[] for _ in nodes]
    to, cap, cost = [], [], []

    def edge(u, v, c, w):
        for a, b, cc, ww in ((u, v, c, w), (v, u, 0, -w)):
            graph[idx[a]].append(len(to))
            to.append(idx[b])
            cap.append(cc)
            cost.append(ww)

    big = sum(max(0, s) for s in stock.values())
    for l in lots:
        edge(S, ("L", l), max(0, int(stock[l])), 0)
    for (l, b), v in value.items():
        if l in stock and b in demand and v > 0:
            edge(("L", l), ("B", b), big, -int(round(v * 100)))
    for b in buyers:
        edge(("B", b), CAP, max(0, int(demand[b])), 0)
    edge(CAP, T, big if total_cap is None else max(0, int(total_cap)), 0)

    n = len(nodes)
    while True:                                          # successive shortest paths (Bellman-Ford, tiny graph)
        dist, prev = [float("inf")] * n, [-1] * n
        dist[idx[S]] = 0
        for _ in range(n):
            changed = False
            for u in range(n):
                if dist[u] == float("inf"):
                    continue
                for e in graph[u]:
                    if cap[e] > 0 and dist[u] + cost[e] < dist[to[e]]:
                        dist[to[e]], prev[to[e]] = dist[u] + cost[e], e
                        changed = True
            if not changed:
                break
        if dist[idx[T]] == float("inf") or dist[idx[T]] >= 0:   # no more profitable path
            break
        f, v = big, idx[T]
        while v != idx[S]:
            e = prev[v]
            f = min(f, cap[e])
            v = to[e ^ 1]
        v = idx[T]
        while v != idx[S]:
            e = prev[v]
            cap[e] -= f
            cap[e ^ 1] += f
            v = to[e ^ 1]

    out: dict[str, dict[str, int]] = {}
    for l in lots:
        for e in graph[idx[("L", l)]]:
            node = nodes[to[e]]
            if isinstance(node, tuple) and node[0] == "B" and e % 2 == 0:
                used = cap[e ^ 1]                        # flow = capacity of the reverse edge
                if used > 0:
                    out.setdefault(l, {})[node[1]] = used
    return out
