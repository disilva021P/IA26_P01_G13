"""
agent.py: o agente completo. Divide o problema em grupos temporais independentes (se existirem), constrói uma solução
inicial com o Greedy, melhora a com Hill Climbing, junta as rotas e valida o plano final com P.audit.
"""
import math

from informed import greedy_chrono, best_first, h_chain, plan_to_routes
from local_search import hill_climbing


def temporal_groups(P, slack=120.0):
    """
    Divide os pedidos em grupos temporalmente independentes (índices em P.req): um pedido abre um grupo novo se a sua
    recolha começa depois do fim da última janela de entrega do grupo atual, mais uma margem (viagem mais longa + slack).
    """
    margin = max(max(row) for row in P.T) + slack
    groups, cur, end = [], [], -math.inf
    for r in sorted(range(P.n), key=lambda r: P.req[r].a):
        q = P.req[r]
        if cur and q.a > end + margin:
            groups.append(cur); cur, end = [], -math.inf
        cur.append(r); end = max(end, q.l)
    return groups + ([cur] if cur else [])


def initial_solution(P, max_nodes=200000, time_limit=60):
    """
    Solução inicial que serve TODOS os pedidos de P (para alimentar o Hill Climbing). Tenta primeiro o Greedy construtivo
    cronológico (rápido) e, só se ele falhar, o A* ponderado (peso 5) com a h1. Devolve (rotas, resultado) ou (None, None).
    """
    res = greedy_chrono(P, max_nodes=max_nodes, time_limit=time_limit)
    if res["found"]:
        return plan_to_routes(P, res["plan"]), res
    res = best_first(P, h_chain, mode="astar", weight=5.0, max_nodes=max_nodes, time_limit=time_limit)
    return (plan_to_routes(P, res["plan"]), res) if res["found"] else (None, None)


def _to_global(P, Q, routes):
    gi = {q.id: i for i, q in enumerate(P.req)}
    return [[(a, None if r is None else gi[Q.req[r].id]) for a, r in rt] for rt in routes]


def solve_all(P, improve=True, hc_time=120, seed=0, **kw):
    """
    Resolve o problema completo por grupos temporais independentes (Greedy e, se falhar, A* ponderado), opcionalmente
    melhora cada grupo com Hill Climbing (até hc_time segundos por grupo), e junta as rotas. Valida no fim com P.audit.
    Devolve (rotas, info por grupo, auditoria).
    """
    routes, info = [[] for _ in range(P.K)], []
    for g in temporal_groups(P):
        Q = P.subset([P.req[r].id for r in g])
        rt, res = initial_solution(Q, **kw)
        if rt is None:
            info.append(dict(ids=[q.id for q in Q.req], found=False)); continue
        c0 = Q.cost(rt)
        hc = None
        if improve:
            hc = hill_climbing(Q, rt, time_limit=hc_time, seed=seed)
            rt = hc["routes"]
        for v, part in enumerate(_to_global(P, Q, rt)):
            routes[v] += part
        info.append(dict(pedidos=Q.n, found=True, custo_inicial=round(c0, 2), custo=round(Q.cost(rt), 2),
                         hc_iter=len(hc["history"]) - 1 if hc else 0, hc_seg=round(hc["seconds"], 1) if hc else 0.0,
                         hc_paragem=hc["stop"] if hc else None))
    return routes, info, P.audit(routes)
