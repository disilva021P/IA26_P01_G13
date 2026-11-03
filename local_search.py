"""
local_search.py: pesquisa local (Hill Climbing) sobre uma solução completa do modelo de model.py.

Uma solução é uma lista de ações por veículo (rotas). O custo de cada vizinho é calculado por P.cost (simulador do modelo),
que devolve infinito se o vizinho violar alguma restrição hard ou deixar pedidos por servir.
"""
import math, random, time

from model import PICK, LOAD, CENTER, DELIVER


def neighbours(routes, rng, P=None):
    """
    Vizinhos (como descrições de movimentos, em ordem aleatória): trocar duas ações; mover uma ação para outra posição ou
    veículo; mover a cadeia completa P, L, D de um pedido para outro veículo ou posição; remover uma visita ao centro (C);
    mover uma paragem inteira (bloco de ações seguidas do mesmo tipo no mesmo local) para outra posição ou veículo;
    mover o par (L, D) de um pedido para uma paragem do armazém de qualquer veículo e uma posição de entrega seguinte.
    """
    K, cand = len(routes), []
    # posições de inserção: só nas fronteiras entre paragens (dentro de uma paragem as posições são equivalentes)
    if P is not None:
        bounds = [sorted({x for i, j in _blocks(P, routes[w]) for x in (i, j)}) for w in range(K)]
    else:
        bounds = [list(range(len(routes[w]) + 1)) for w in range(K)]
    for v in range(K):
        R = routes[v]
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                cand.append(("swap", v, i, j))
            for w in range(K):
                for j in bounds[w]:
                    if w == v and j in (i, i + 1):
                        continue
                    cand.append(("move", v, i, w, j - 1 if (w == v and j > i) else j))
            if R[i][0] == CENTER:
                cand.append(("drop", v, i))
    for r in sorted({a[1] for R in routes for a in R if a[1] is not None}):
        for w in range(K):
            for j in bounds[w]:
                cand.append(("chain", r, w, j))
    if P is None:
        rng.shuffle(cand)
        return cand
    # blocos: ações seguidas do mesmo tipo e no mesmo local (uma paragem) movidas em conjunto
    for v in range(K):
        for i, j in _blocks(P, routes[v]):
            if j - i < 2:
                continue
            for w in range(K):
                for k in bounds[w]:
                    if not (w == v and i <= k <= j):
                        cand.append(("block", v, i, j, w, k))
    # par L, D de um pedido: L entra numa paragem do armazém (início ou fim de um bloco) e D numa fronteira seguinte
    reqs = sorted({a[1] for R in routes for a in R if a[0] == LOAD})
    for w in range(K):
        wh = sorted({x for i, j in _blocks(P, routes[w]) if routes[w][i][0] in (LOAD, CENTER) for x in (i, j)})
        for r in reqs:
            if w not in P.allowed[r]:
                continue
            for j in wh:
                for k in bounds[w]:
                    if k >= j:
                        cand.append(("pair", r, w, j, k))
    rng.shuffle(cand)
    return cand


def _blocks(P, R):
    """Intervalos [i, j) de ações seguidas do mesmo tipo no mesmo local (uma paragem com vários artigos)."""
    def key(a):
        if a[0] == PICK: return (PICK, P.req[a[1]].p)
        if a[0] == DELIVER: return (DELIVER, P.req[a[1]].d)
        return ("W", P.C)
    out, i = [], 0
    while i < len(R):
        j = i + 1
        while j < len(R) and key(R[j]) == key(R[i]):
            j += 1
        out.append((i, j)); i = j
    return out


def apply_move(routes, m):
    """Aplica o movimento m e devolve as novas rotas (as originais não são alteradas)."""
    R = [list(x) for x in routes]
    if m[0] == "swap":
        _, v, i, j = m
        R[v][i], R[v][j] = R[v][j], R[v][i]
    elif m[0] == "move":
        _, v, i, w, j = m
        R[w].insert(j, R[v].pop(i))
    elif m[0] == "block":
        _, v, i, j, w, k = m
        seg = R[v][i:j]
        del R[v][i:j]
        if w == v and k > j:
            k -= j - i
        R[w][k:k] = seg
    elif m[0] == "pair":
        _, r, w, j, k = m
        # j e k são posições na rota w ANTES de retirar L r e D r: marca se as antigas e só no fim se apagam
        for v in range(len(R)):
            R[v] = [a if not (a[1] == r and a[0] in (LOAD, DELIVER)) else ("X", r) for a in R[v]]
        R[w].insert(k, (DELIVER, r))
        R[w].insert(j, (LOAD, r))
        for v in range(len(R)):
            R[v] = [a for a in R[v] if a[0] != "X"]
    elif m[0] == "drop":
        _, v, i = m
        R[v].pop(i)
    else:
        _, r, w, j = m
        chain = []
        for v in range(len(R)):
            keep = []
            for a in R[v]:
                (chain if a[1] == r else keep).append(a)
            R[v] = keep
        chain.sort(key=lambda a: {PICK: 0, LOAD: 1, DELIVER: 2}[a[0]])
        j = min(j, len(R[w]))
        R[w][j:j] = chain
    return R


def hill_climbing(P, routes, max_iter=500, seed=0, time_limit=120):
    """
    Hill Climbing com primeira melhoria: percorre os vizinhos por ordem aleatória (semente) e aceita o primeiro viável com
    custo estritamente menor; pára num máximo local, ao fim de max_iter iterações ou de time_limit segundos.
    Devolve rotas, custo, histórico do custo por iteração (history), instante de cada iteração (times), número de
    vizinhos avaliados (evals) e o motivo de paragem (stop).
    """
    rng, t0 = random.Random(seed), time.time()
    cur = [list(r) for r in routes]
    cost = P.cost(cur)
    if math.isinf(cost):
        raise ValueError("A solução inicial é inviável ou não serve todos os pedidos (o Hill Climbing não insere pedidos em falta).")
    history, times, evals, stop = [cost], [0.0], 0, "max_iter"
    for _ in range(max_iter):
        improved = False
        for m in neighbours(cur, rng, P):
            if time.time() - t0 > time_limit:
                break
            cand = apply_move(cur, m)
            evals += 1
            c = P.cost(cand)
            if c < cost - 1e-6:
                cur, cost, improved = cand, c, True
                break
        history.append(cost)
        times.append(time.time() - t0)
        if time.time() - t0 > time_limit:
            stop = "tempo"; break
        if not improved:
            stop = "ótimo local"; break
    return dict(routes=cur, cost=cost, history=history, times=times, evals=evals, seconds=time.time() - t0, stop=stop)
