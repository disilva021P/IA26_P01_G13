"""
algoritmos.py — Algoritmos do projeto "Comida Local" (Trabalho de IA, UPCA 2026-27), sobre o modelo de model.py.

  1. Pesquisa informada : Greedy e A* (best_first) com duas heurísticas admissíveis (h_chain, h_disjoint)
  2. Pesquisa local     : Hill Climbing sobre uma solução inicial (hill_climbing)
  3. CSP                : paragens (P, C, L, D) como variáveis; backtracking com MRV, forward checking, restrição de capacidade e
                          heurística de valores para a restrição soft (csp_solve)

Custo = distância (km) + LATE_PENALTY * minutos de atraso (ver model.py). Uma solução é uma lista de ações por veículo (rotas).
"""
import heapq, itertools, math, random, time
from dataclasses import dataclass

import numpy as np

from model import (PICK, LOAD, CENTER, DELIVER, BLOCKED, EPS, SERVICE_MIN, RETURN_TO_CENTER)

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
# 3. CSP
# ======================================================================================================================
# Formulação: 4 variáveis por pedido, uma por PARAGEM (evento), e não uma por viagem:
#     P_r  recolha no produtor | C_r  chegada ao armazém do veículo que recolheu (descarga) |
#     L_r  carga no armazém pelo veículo que vai entregar | D_r  entrega no cliente
# Domínio de cada variável = (veículo, instante de início do serviço), com o tempo discretizado em SLOT minutos.
# O veículo que recolhe (P_r, C_r) pode ser diferente do que entrega (L_r, D_r): é a passagem de testemunho pelo armazém do
# modelo. Quando é o mesmo, a descarga é feita pela própria L_r e C_r toma o valor "virtual" (veículo, None): não é paragem.
# As viagens com vários pedidos (P x, P y, L x, L y, D x, D y) NÃO são impostas nem pós-processadas: aparecem sozinhas quando
# paragens de pedidos diferentes ficam seguidas no mesmo veículo, porque o custo é a distância entre paragens consecutivas.
SLOT = 5           # granularidade temporal das variáveis (min); o custo final é sempre recalculado por P.simulate (horário mais cedo)
MAX_LATE = 0       # folga de atraso (min) permitida nos domínios, além do atraso inevitável de cada pedido (soft, com penalização)
KINDS = (PICK, CENTER, LOAD, DELIVER)              # ordem das 4 variáveis de cada pedido: índice da variável = 4 * r + posição
KIND_RANK = {k: i for i, k in enumerate(KINDS)}

@dataclass
class Job:
    """Variável do CSP: paragem `kind` (P, C, L ou D) do pedido r, no local `loc`. Domínio: lista de pares (veículo, hora)."""
    idx: int; r: int; kind: str; loc: int; domain: list; due: float = None

def _grid(lo, hi, slot):
    return [float(t) for t in np.arange(math.ceil(lo / slot - EPS) * slot, hi + EPS, slot)]

def build_jobs(P, max_late=MAX_LATE, slot=SLOT):
    """
    4 variáveis por pedido. Os limites dos domínios vêm das restrições unárias e da propagação das precedências
    P -> C -> L -> D (tempos mínimos de viagem + serviço), por isso já nascem podados.
      P: [availability_start, availability_end]                          (hard)
      D: [delivery_start, delivery_end + atraso inevitável + max_late]   (início hard; fim soft, com penalização)
         O atraso inevitável (P.lateness_best_case) é o do melhor caso do pedido, por isso `max_late` é só a folga EXTRA.
      C, L: o que sobra entre os dois. C tem ainda o valor virtual (veículo, None).
    """
    S, T, C, jobs = SERVICE_MIN, P.T, P.C, []
    for r, q in enumerate(P.req):
        vs = [v for v in range(P.K) if P.w[r] <= P.cap[v] + EPS]                   # restrição unária: capacidade
        d_pl, d_ld = S + T[q.p][C], S + T[C][q.d]                                  # P->C e L->D (serviço + viagem)
        l_lo = q.a + d_pl
        d_lo = max(q.e, l_lo + d_ld)
        d_hi = q.l + math.ceil(P.lateness_best_case(q) / slot) * slot + max_late + slot
        l_hi = d_hi - d_ld
        p_hi = min(q.a_end, l_hi - d_pl)
        real = lambda lo, hi: [(v, t) for t in _grid(lo, hi, slot) for v in vs]
        doms = (real(q.a, p_hi), [(v, None) for v in vs] + real(l_lo, l_hi), real(l_lo, l_hi), real(d_lo, d_hi))
        for kind, loc, dom in zip(KINDS, (q.p, C, C, q.d), doms):
            jobs.append(Job(len(jobs), r, kind, loc, dom, q.l if kind == DELIVER else None))
    return jobs

def csp_compatible(P, ji, vi, ti, jj, vj, tj):
    """
    Restrições binárias (ti/tj = None só na descarga virtual).
      mesmo pedido : P e C no mesmo veículo e C depois da viagem; C virtual => L no mesmo veículo da recolha;
                     C real => carga só depois de os produtos chegarem ao armazém; L e D no mesmo veículo e D depois da viagem;
                     se o mesmo veículo faz P e L (ou P e D), respeita-se o tempo de viagem entre eles;
      mesmo veículo: duas paragens não se sobrepõem (serviço + viagem entre os locais), em qualquer ordem.
    """
    S, T, C = SERVICE_MIN, P.T, P.C
    if ji.r == jj.r:
        if KIND_RANK[ji.kind] > KIND_RANK[jj.kind]:
            ji, vi, ti, jj, vj, tj = jj, vj, tj, ji, vi, ti
        pair = (ji.kind, jj.kind)
        if pair == (PICK, CENTER):
            if vi != vj or (tj is not None and tj < ti + S + T[ji.loc][C] - EPS):
                return False
        elif pair == (PICK, LOAD):
            if vi == vj and tj < ti + S + T[ji.loc][C] - EPS:
                return False
        elif pair == (PICK, DELIVER):
            if vi == vj and tj < ti + 2 * S + T[ji.loc][C] + T[C][jj.loc] - EPS:
                return False
        elif pair == (CENTER, LOAD):
            if (vi != vj) if ti is None else (tj < ti - EPS):
                return False
        elif pair == (LOAD, DELIVER):
            if vi != vj or tj < ti + S + T[C][jj.loc] - EPS:
                return False
    if ti is None or tj is None or vi != vj:
        return True
    return ti + S + T[ji.loc][jj.loc] <= tj + EPS or tj + S + T[jj.loc][ji.loc] <= ti + EPS

def csp_neighbors(P, jobs):
    """Só ligam variáveis cujos intervalos possam colidir (pedidos de dias diferentes são independentes)."""
    span = []
    for j in jobs:
        ts = [t for _, t in j.domain if t is not None]
        span.append((min(ts), max(ts) + SERVICE_MIN))
    mt = max(max(row) for row in P.T) + SERVICE_MIN
    nb = {j.idx: [] for j in jobs}
    for i in jobs:
        for j in jobs:
            if i.idx < j.idx and (i.r == j.r or (span[i.idx][0] < span[j.idx][1] + mt and span[j.idx][0] < span[i.idx][1] + mt)):
                nb[i.idx].append(j.idx); nb[j.idx].append(i.idx)
    return nb

def capacity_ok(P, jobs, assigned, v):
    """
    Restrição global de capacidade do veículo v. Um pedido está a bordo do veículo que recolhe desde P até à descarga (C real, ou
    a entrega se a descarga é virtual) e a bordo do que entrega desde L até D (limite conservador: ignora o tempo no armazém).
    Só conta intervalos já conhecidos, por isso nunca poda uma atribuição que ainda possa ser completada.
    """
    iv = []
    for r in range(P.n):
        g = lambda k: assigned.get(4 * r + k)
        p, c, l, d = g(0), g(1), g(2), g(3)
        if p and p[0] == v:
            end = (c[1] if c[1] is not None else (d[1] if d and d[0] == v else None)) if c else None
            iv.append((p[1], end, P.w[r]))
        if l and l[0] == v:
            iv.append((l[1], d[1] if d else None, P.w[r]))
    return all(sum(w for s2, e2, w in iv if s2 == s or (s2 < s and e2 is not None and e2 > s)) <= P.cap[v] + EPS for s, _, _ in iv)

def _late(P, j, t):
    return P.late_penalty * max(0.0, t - j.due) if j.kind == DELIVER else 0.0

def _ordered(jobs, assigned, v):
    """Paragens reais do veículo v por ordem temporal (a descarga virtual não é paragem)."""
    return sorted((t, KIND_RANK[jobs[i].kind], jobs[i].r, i) for i, (vk, t) in assigned.items() if vk == v and t is not None)

def csp_cost(P, jobs, assigned):
    """
    Custo planeado (km + penalização de atrasos) de uma atribuição parcial ou completa: por veículo, centro -> paragens por
    ordem temporal -> centro. Inserir uma paragem nunca reduz a distância (desigualdade triangular), logo o custo parcial é
    um limite inferior do custo de qualquer completação, e serve para podar no branch & bound.
    """
    tot = 0.0
    for v in range(P.K):
        loc = P.veh[v].center
        for t, _, _, i in _ordered(jobs, assigned, v):
            tot += P.D[loc][jobs[i].loc] + _late(P, jobs[i], t)
            loc = jobs[i].loc
        tot += P.D[loc][P.veh[v].center]
    return tot

def incremental_cost(P, jobs, assigned, j, v, t):
    """Heurística de valores (restrição soft): custo extra de inserir (v, t) entre a paragem anterior e a seguinte do veículo."""
    if t is None:
        return 0.0                                                   # descarga virtual: não acrescenta paragem
    pred = succ = None
    for k, (vk, tk) in assigned.items():
        if vk != v or tk is None:
            continue
        if tk <= t and (pred is None or tk > assigned[pred][1]): pred = k
        if tk > t and (succ is None or tk < assigned[succ][1]): succ = k
    a = jobs[pred].loc if pred is not None else P.veh[v].center
    b = jobs[succ].loc if succ is not None else P.veh[v].center
    return P.D[a][j.loc] + P.D[j.loc][b] - P.D[a][b] + _late(P, j, t)

def csp_to_routes(P, jobs, assigned):
    """Ordena as paragens de cada veículo pelo instante e traduz para ações do modelo (P r, C, L r, D r)."""
    routes = [[] for _ in range(P.K)]
    for v in range(P.K):
        for _, _, _, i in _ordered(jobs, assigned, v):
            act = (jobs[i].kind, None if jobs[i].kind == CENTER else jobs[i].r)
            if act != (CENTER, None) or not routes[v] or routes[v][-1] != act:      # uma só descarga para várias seguidas
                routes[v].append(act)
    return routes

class _Restart(Exception):
    pass

def csp_solve(P, var_order="static", value_order="lex", forward_checking=False, max_nodes=300000, time_limit=60, optimize=False, max_late=MAX_LATE, slot=SLOT,
              restarts=0, restart_nodes=2000, seed=0):
    """
    Backtracking. var_order: "static" | "mrv" (menos valores legais, desempate por grau). value_order: "lex" | "soft" (menor custo extra).
    forward_checking: filtra os domínios dos vizinhos a cada atribuição. optimize=True: depois da 1ª solução continua em
    branch & bound (poda pelo custo parcial) até max_nodes/time_limit. Conta nós e backtracks.
    Devolve também `timed_out` e `complete` (True só se o branch & bound esgotou a árvore: o custo planeado é então ótimo
    para esta formulação). O custo final (`cost`) é recalculado por P.simulate, que executa cada rota o mais cedo possível,
    por isso é <= ao custo planeado (`cost_planned`), que usa os instantes discretizados.
    restarts > 0: reinícios aleatórios (contra o thrashing do backtracking cronológico). Cada tentativa tem um orçamento de
    `restart_nodes` nós (que cresce 1.5x a cada reinício); a partir da 2ª, os desempates das variáveis e os valores são perturbados.
    """
    jobs, t0 = build_jobs(P, max_late, slot), time.time()
    out = dict(found=False, first=None, nodes=0, backtracks=0, seconds=0.0, timed_out=False, complete=False,
               cost=None, km=None, late_min=None, cost_planned=None, routes=None)
    if any(not j.domain for j in jobs):                    # domínio vazio logo à partida: pedido impossível com este max_late
        return out
    nb = csp_neighbors(P, jobs)
    dom = {j.idx: list(j.domain) for j in jobs}
    assigned, st = {}, dict(nodes=0, backtracks=0)
    best = dict(cost=math.inf, assigned=None, first=None)
    tkey = lambda x: -1.0 if x[1] is None else x[1]
    rng, attempt = random.Random(seed), dict(k=0, budget=math.inf, nodes0=0)
    JITTER = 30.0                                           # perturbação máxima (km-equivalentes) dos valores nos reinícios

    def select():
        free = [j.idx for j in jobs if j.idx not in assigned]
        if var_order == "static":
            return free[0]
        return min(free, key=lambda i: (len(dom[i]), -len(nb[i]), rng.random() if attempt["k"] else 0))

    def values(i):
        noise = (lambda: rng.uniform(0, JITTER)) if attempt["k"] else (lambda: 0.0)
        if value_order == "soft":
            return sorted(dom[i], key=lambda x: (incremental_cost(P, jobs, assigned, jobs[i], x[0], x[1]) + noise(), tkey(x)))
        return sorted(dom[i], key=lambda x: (x[0], tkey(x)))

    def rec():
        if len(assigned) == len(jobs):
            c = csp_cost(P, jobs, assigned)
            if c < best["cost"] - 1e-6:
                best.update(cost=c, assigned=dict(assigned))
                if best["first"] is None:
                    best["first"] = (c, st["nodes"], st["backtracks"])
            return not optimize
        if optimize and best["assigned"] is not None and csp_cost(P, jobs, assigned) >= best["cost"] - 1e-6:
            return False
        if st["nodes"] > max_nodes or time.time() - t0 > time_limit:
            raise TimeoutError
        if st["nodes"] - attempt["nodes0"] > attempt["budget"]:
            raise _Restart
        i = select()
        for (v, t) in values(i):
            st["nodes"] += 1
            if any(k in assigned and not csp_compatible(P, jobs[i], v, t, jobs[k], *assigned[k]) for k in nb[i]):
                st["backtracks"] += 1
                continue
            assigned[i] = (v, t)
            if not capacity_ok(P, jobs, assigned, v):
                del assigned[i]; st["backtracks"] += 1
                continue
            trail, ok = [], True
            if forward_checking:
                for k in nb[i]:
                    if k in assigned:
                        continue
                    new = [x for x in dom[k] if csp_compatible(P, jobs[i], v, t, jobs[k], *x)]
                    if len(new) < len(dom[k]):
                        trail.append((k, dom[k])); dom[k] = new
                    if not new:
                        ok = False; break
            if ok and rec():
                return True
            for k, old in trail:
                dom[k] = old
            del assigned[i]
            st["backtracks"] += 1
        return False

    try:
        for k in range(restarts + 1):
            attempt.update(k=k, nodes0=st["nodes"], budget=restart_nodes * 1.5 ** k if k < restarts else math.inf)
            assigned.clear()
            dom.update({j.idx: list(j.domain) for j in jobs})
            try:
                rec()
                break                                       # a pesquisa terminou (solução, ou árvore esgotada): não há reinício
            except _Restart:
                continue
    except TimeoutError:
        out["timed_out"] = True
    out.update(found=best["assigned"] is not None, first=best["first"], nodes=st["nodes"], backtracks=st["backtracks"],
               seconds=time.time() - t0, complete=optimize and not out["timed_out"])
    if out["found"]:
        out["routes"] = csp_to_routes(P, jobs, best["assigned"])
        out["cost_planned"] = best["cost"]
        res = P.simulate(out["routes"])
        if res:
            out.update(cost=res["cost"], km=res["km"], late_min=res["late_min"])
    return out

def csp_complete(P, limits=(0, 15, 30, 60, 120, 480, 1440), **kw):
    """
    Relaxamento gradual da restrição soft: começa por só permitir o atraso inevitável de cada pedido e alarga a folga de atraso
    dos domínios (`limits`, em min) até haver solução. Com a penalização alta, as boas soluções quase não têm atraso evitável.
    """
    kw.setdefault("restarts", 12)
    for L in limits:
        out = csp_solve(P, max_late=L, **kw)
        if out["found"]:
            out["max_late_used"] = L
            return out
    return out

# ======================================================================================================================
# 4. GARANTIR QUE TODOS OS PEDIDOS SÃO ENTREGUES
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

def _to_global(P, Q, routes):
    gi = {q.id: i for i, q in enumerate(P.req)}
    return [[(a, None if r is None else gi[Q.req[r].id]) for a, r in rt] for rt in routes]

def solve_all(P, improve=True, **kw):
    """
    Resolve o problema completo por grupos temporais independentes (A* -> A* ponderado como fallback), opcionalmente
    melhora cada grupo com Hill Climbing, e junta as rotas. Valida no fim com P.audit (todos os pedidos servidos).
    """
    routes, info = [[] for _ in range(P.K)], []
    for g in temporal_groups(P):
        Q = P.subset([P.req[r].id for r in g])
        rt, res = initial_solution(Q, **kw)
        if rt is None:
            info.append(dict(ids=[q.id for q in Q.req], found=False)); continue
        if improve:
            rt = hill_climbing(Q, rt)["routes"]
        for v, part in enumerate(_to_global(P, Q, rt)):
            routes[v] += part
        info.append(dict(ids=[q.id for q in Q.req], found=True, cost=Q.cost(rt)))
    return routes, info, P.audit(routes)
