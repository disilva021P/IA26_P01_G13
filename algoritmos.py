"""
algoritmos.py — Algoritmos do projeto "Comida Local" (Trabalho de IA, UPCA 2026-27), sobre o modelo de model.py.

  1. Pesquisa informada : Greedy e A* (best_first) com duas heurísticas admissíveis (h_chain, h_disjoint)
<<<<<<< HEAD
  2. Pesquisa local     : Hill Climbing sobre uma solução inicial (hill_climbing)

Custo = distância (km) + LATE_PENALTY * minutos de atraso (ver model.py). Uma solução é uma lista de ações por veículo (rotas).
"""
import heapq, itertools, math, random, time

from model import PICK, LOAD, CENTER, DELIVER, BLOCKED, EPS
=======

Custo = distância (km) + LATE_PENALTY * minutos de atraso. Uma solução é uma lista de ações por veículo (rotas).
"""
import heapq, itertools, math, time

from model import BLOCKED, EPS
>>>>>>> 2c8d1b1a47c1f52fa4e6bf3ef33a47cedc5e08fb

# ======================================================================================================================
# 1. PESQUISA INFORMADA
# ======================================================================================================================
def is_dead(P, s):
    """Estado sem saída: algum pedido pendente já não consegue ser recolhido dentro da janela de recolha (restrição hard)."""
    for r in range(P.n):
        if s.status[r] != P.PENDING:
            continue
        q = P.req[r]
        if all(t + P.T[loc][q.p] > q.a_end + EPS for loc, t, _ in s.veh):
            return True
    return False

def late_bound(P, s):
    """Atraso (min) inevitável dos pedidos por entregar, no melhor caso a partir de s (limite inferior da parte soft do custo)."""
    D, T, C, late = P.D, P.T, P.C, 0.0
    for r in range(P.n):
        c, q = s.status[r], P.req[r]
        if c == P.DONE:
            continue
        if c == P.PENDING:
            tw = min(max(t + T[loc][q.p], q.a) + T[q.p][C] for loc, t, _ in s.veh)
        elif 1 <= c <= P.K:
            loc, t, _ = s.veh[c - 1]
            tw = t + T[loc][C]
        elif c == P.WH:
            tw = min(max(t + T[loc][C], s.wt[r]) for loc, t, _ in s.veh)
        else:
            loc, t, _ = s.veh[c - P.ONB]
            late += max(0.0, max(t + T[loc][q.d], q.e) - q.l)
            continue
        late += max(0.0, max(tw + T[C][q.d], q.e) - q.l)
    return late

def h_chain(P, s):
    """
    Heurística 1 — cadeia mais longa (admissível). Cada pedido pendente obriga a percorrer produtor -> centro -> cliente,
    por isso a distância total nunca é inferior à do pedido mais exigente. Soma-se o atraso inevitável (parte soft).
    """
    D, C, best = P.D, P.C, 0.0
    for r in range(P.n):
        c, q = s.status[r], P.req[r]
        if c == P.DONE:
            continue
        if c == P.PENDING:
            x = min(D[loc][q.p] for loc, _, _ in s.veh) + D[q.p][C] + D[C][q.d]
        elif 1 <= c <= P.K:
            x = D[s.veh[c - 1][0]][C] + D[C][q.d]
        elif c == P.WH:
            x = min(D[loc][C] for loc, _, _ in s.veh) + D[C][q.d]
        else:
            x = D[s.veh[c - P.ONB][0]][q.d]
        best = max(best, x)
    return best + P.late_penalty * late_bound(P, s)

def h_disjoint(P, s):
    """
    Heurística 2 — janelas disjuntas (admissível). Dois pedidos cujas janelas [início da recolha, fim da entrega] não se
    sobrepõem nunca viajam juntos, logo os troços "com carga" de cada um somam-se. Escolhe-se, por programação dinâmica
    (interval scheduling pesado), o conjunto de pedidos pendentes com janelas disjuntas que maximiza essa soma; toma-se o
    máximo com h_chain. Soma-se o atraso inevitável.
    """
    D, C, items = P.D, P.C, []
    for r in range(P.n):
        c, q = s.status[r], P.req[r]
        if c == P.DONE:
            continue
        if c == P.PENDING:
            w = D[q.p][C] + D[C][q.d]
        elif 1 <= c <= P.K:
            w = D[s.veh[c - 1][0]][C] + D[C][q.d]
        elif c == P.WH:
            w = D[C][q.d]
        else:
            w = D[s.veh[c - P.ONB][0]][q.d]
        items.append((q.l, q.a, w))
    items.sort()
    best = [0.0] * (len(items) + 1)
    for i, (l, a, w) in enumerate(items, 1):
        j = i - 1
        while j > 0 and items[j - 1][0] >= a:
            j -= 1
        best[i] = max(best[i - 1], w + best[j])
    km = max(best[-1], h_chain(P, s) - P.late_penalty * late_bound(P, s))
    return km + P.late_penalty * late_bound(P, s)

HEURISTICS = {"h1_cadeia": h_chain, "h2_disjuntas": h_disjoint}

def plan_to_routes(P, plan):
    """Converte um plano [(veículo, ação), ...] em rotas: uma lista de ações por veículo."""
    routes = [[] for _ in range(P.K)]
    for v, act in plan:
        routes[v].append(act)
    return routes

def best_first(P, h, mode="astar", weight=1.0, max_nodes=200000, time_limit=60, use_dominance=True):
    """
    Pesquisa best-first no espaço de estados. mode="greedy": f = h ; mode="astar": f = g + weight*h (weight>1 = A* ponderado).
    Cortes: estados mortos (janela de recolha) e dominância (mesmo estado físico com custo e tempos <= ).
    Devolve dict(found, expanded, generated, seconds, cost, km, late_min, plan).
    """
    t0, s0, cnt = time.time(), P.initial_state(), itertools.count()
    prio = lambda g, hv: hv if mode == "greedy" else g + weight * hv
    openh = [(prio(0.0, h(P, s0)), next(cnt), 0.0, s0, False)]
    best_g, parent, seen = {s0: 0.0}, {s0: None}, {}
    expanded = generated = 0
    final = None

    def dominated(s2, g2):
        pk = (tuple(x[0] for x in s2.veh), s2.status)
        vec = tuple(x[1] for x in s2.veh) + s2.wt
        lst = seen.setdefault(pk, [])
        for g_, v_ in lst:
            if g_ <= g2 + EPS and all(a <= b + EPS for a, b in zip(v_, vec)):
                return True
        lst[:] = [(g_, v_) for g_, v_ in lst if not (g2 <= g_ + EPS and all(b <= a + EPS for a, b in zip(v_, vec)))]
        lst.append((g2, vec))
        return False

    while openh:
        f, _, g, s, fin = heapq.heappop(openh)
        if fin:
            final = (g, s); break
        if g > best_g.get(s, math.inf) + EPS:
            continue
        expanded += 1
        if expanded > max_nodes or time.time() - t0 > time_limit:
            break
        if P.is_goal(s):                                    # objetivo: acrescenta-se o regresso ao centro como passo final
            g_end = g + P.return_cost(s)
            heapq.heappush(openh, (g_end if mode == "astar" else -1.0, next(cnt), g_end, s, True))
            continue
        for v in range(P.K):
            for act in P.actions(s, v):
                res = P.apply(s, v, act)
                if res is None or res is BLOCKED:
                    continue
                s2, km, late = res
                if is_dead(P, s2):
                    continue
                g2 = g + P.step_cost(km, late)
                if use_dominance and dominated(s2, g2):
                    continue
                if g2 < best_g.get(s2, math.inf) - EPS:
                    best_g[s2] = g2; parent[s2] = (s, v, act); generated += 1
                    heapq.heappush(openh, (prio(g2, h(P, s2)), next(cnt), g2, s2, False))
    out = dict(found=final is not None, expanded=expanded, generated=generated, seconds=time.time() - t0,
               cost=None, km=None, late_min=None, plan=None)
    if final:
        s, plan = final[1], []
        while parent[s] is not None:
            s, v, act = parent[s]; plan.append((v, act))
        plan.reverse()
        res = P.simulate(plan_to_routes(P, plan))            # custo recalculado pelo simulador do modelo
        out.update(plan=plan, cost=res["cost"], km=res["km"], late_min=res["late_min"])
    return out
<<<<<<< HEAD

# ======================================================================================================================
# 2. PESQUISA LOCAL — HILL CLIMBING
# ======================================================================================================================
def neighbours(routes, rng):
    """
    Vizinhos (como descrições de movimentos, em ordem aleatória): trocar duas ações; mover uma ação para outra posição/veículo;
    mover a cadeia completa P-L-D de um pedido para outro veículo/posição; remover uma visita ao centro (C).
    """
    K, cand = len(routes), []
    for v in range(K):
        R = routes[v]
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                cand.append(("swap", v, i, j))
            for w in range(K):
                for j in range(len(routes[w]) + (1 if w != v else 0)):
                    if not (w == v and j == i):
                        cand.append(("move", v, i, w, j))
            if R[i][0] == CENTER:
                cand.append(("drop", v, i))
    for r in sorted({a[1] for R in routes for a in R if a[1] is not None}):
        for w in range(K):
            for j in range(len(routes[w]) + 1):
                cand.append(("chain", r, w, j))
    rng.shuffle(cand)
    return cand

def apply_move(routes, m):
    R = [list(x) for x in routes]
    if m[0] == "swap":
        _, v, i, j = m
        R[v][i], R[v][j] = R[v][j], R[v][i]
    elif m[0] == "move":
        _, v, i, w, j = m
        R[w].insert(j, R[v].pop(i))
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

def hill_climbing(P, routes, max_iter=500, seed=0, time_limit=300):
    """
    Hill Climbing com primeira melhoria: percorre os vizinhos por ordem aleatória (semente) e aceita o primeiro viável com
    custo estritamente menor; pára num máximo local. Devolve rotas, custo e o histórico do custo por iteração.
    """
    rng, t0 = random.Random(seed), time.time()
    cur = [list(r) for r in routes]
    cost = P.cost(cur)
    if math.isinf(cost):
        raise ValueError("A solução inicial é inviável ou não serve todos os pedidos (o Hill Climbing não consegue inserir pedidos em falta).")
    history, evals = [cost], 0
    for _ in range(max_iter):
        improved = False
        for m in neighbours(cur, rng):
            cand = apply_move(cur, m)
            evals += 1
            c = P.cost(cand)
            if c < cost - 1e-6:
                cur, cost, improved = cand, c, True
                break
            if time.time() - t0 > time_limit:
                break
        history.append(cost)
        if not improved or time.time() - t0 > time_limit:
            break
    return dict(routes=cur, cost=cost, history=history, evals=evals, seconds=time.time() - t0)

# ======================================================================================================================
# 3. GARANTIR QUE TODOS OS PEDIDOS SÃO ENTREGUES
# ======================================================================================================================
def temporal_groups(P, slack=120.0):
    """Divide os pedidos em grupos temporalmente independentes (índices em P.req). `slack` = folga (min) para atrasos/regresso."""
    margin = max(max(row) for row in P.T) + slack
    groups, cur, end = [], [], -math.inf
    for r in sorted(range(P.n), key=lambda r: P.req[r].a):
        q = P.req[r]
        if cur and q.a > end + margin:
            groups.append(cur); cur, end = [], -math.inf
        cur.append(r); end = max(end, q.l)
    return groups + ([cur] if cur else [])

def initial_solution(P, h=h_chain, weights=(1.0, 2.0, 5.0, 20.0), max_nodes=200000, time_limit=60):
    """
    Solução inicial que serve TODOS os pedidos de P (para alimentar o Hill Climbing): A* e, se necessário, A* ponderado.
    Devolve (rotas, resultado_da_pesquisa) ou (None, None). Como a recolha/entrega só falha por janelas hard, qualquer
    estado final encontrado serve todos os pedidos; o atraso face a delivery_end é apenas penalizado.
    """
    for w in weights:
        res = best_first(P, h, mode="astar", weight=w, max_nodes=max_nodes, time_limit=time_limit)
        if res["found"]:
            return plan_to_routes(P, res["plan"]), res
    res = best_first(P, h, mode="greedy", max_nodes=max_nodes * 5, time_limit=time_limit * 2)
    return (plan_to_routes(P, res["plan"]), res) if res["found"] else (None, None)
=======
>>>>>>> 2c8d1b1a47c1f52fa4e6bf3ef33a47cedc5e08fb
