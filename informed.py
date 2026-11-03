"""
informed.py: pesquisa informada (Greedy, A* e A* ponderado) sobre o modelo de model.py.

Heurísticas
  h1 (h_chain)    : admissível. Cadeia produtor, armazém, cliente do pedido mais exigente + atraso inevitável.
  h2 (h_disjoint) : máximo entre h1 e a soma dos troços "com carga" de pedidos com janelas disjuntas + atraso inevitável.
                    Admissível enquanto nenhum pedido se atrasa (ver notebook).
  h3 (h_work)     : "trabalho restante", NÃO admissível, usada só pelo Greedy: soma dos troços em falta + LAMBDA × etapas
                    em falta + atraso inevitável. Com LAMBDA grande o Greedy dá prioridade ao progresso (cada ação que
                    avança uma etapa baixa h3 em LAMBDA), e os empates são desfeitos pelo instante da ação.

Custo = distância (km) + LATE_PENALTY * minutos de atraso (ver model.py). Uma solução é uma lista de ações por veículo.
"""
import heapq, itertools, math, time

from model import BLOCKED, EPS

LAMBDA = 1000.0     # peso das etapas em falta na h3


# ----------------------------------------------------------------------------------------------------------------------
# Limites inferiores de tempo
# ----------------------------------------------------------------------------------------------------------------------
def _reach(P, s, v, target):
    """Instante mais cedo (limite inferior) em que o veículo v pode começar um serviço em target."""
    loc, t, _, ss = s.veh[v]
    if loc == target and ss is not None:
        return ss                      # pode juntar se à paragem em curso
    return t + P.T[loc][target]


def _depart_wh(P, s, r):
    """Instante mais cedo (limite inferior) em que o pedido r, já no armazém, pode sair de lá a bordo de um veículo."""
    best = math.inf
    for v in P.allowed[r]:
        loc, t, _, _ = s.veh[v]
        dep = t if loc == P.veh[v].center else t + P.T[loc][P.veh[v].center] + P.H
        best = min(best, max(dep, s.wt[r]))
    return best


def _delivery_lb(P, s, r):
    """Limite inferior do início da entrega do pedido r (antes de esperar por delivery_start)."""
    c, q, T, C, S, H, off = s.status[r], P.req[r], P.T, P.C, P.S, P.H, P.off
    if c == P.PENDING:
        ps = min(max(_reach(P, s, v, q.p), q.a - off) for v in P.allowed[r])
        return ps + S + T[q.p][C] + H + T[C][q.d]
    if 1 <= c <= P.K:
        loc, t, _, _ = s.veh[c - 1]
        return t + T[loc][C] + H + T[C][q.d]
    if c == P.WH:
        return _depart_wh(P, s, r) + T[C][q.d]
    return _reach(P, s, c - P.ONB, q.d)


def is_dead(P, s):
    """
    Estado sem saída: algum pedido pendente já não consegue ser recolhido dentro da janela (por nenhum veículo permitido),
    ou algum pedido no armazém já não consegue ser carregado dentro do tempo máximo no armazém.
    """
    off = P.off
    for r in range(P.n):
        c = s.status[r]
        if c == P.PENDING:
            q = P.req[r]
            if all(_reach(P, s, v, q.p) > q.a_end - off + EPS for v in P.allowed[r]):
                return True
        elif c == P.WH and P.dwell < math.inf:
            limit = s.wt[r] + P.dwell
            ok = False
            for v in P.allowed[r]:
                loc, t, _, ss = s.veh[v]
                C = P.veh[v].center
                start = ss if (loc == C and ss is not None) else t + P.T[loc][C]
                if start <= limit + EPS:
                    ok = True; break
            if not ok:
                return True
    return False


def late_bound(P, s):
    """Atraso (min) inevitável dos pedidos por entregar, a partir de s (limite inferior da parte soft do custo)."""
    late = 0.0
    for r in range(P.n):
        if s.status[r] == P.DONE:
            continue
        q = P.req[r]
        late += max(0.0, max(_delivery_lb(P, s, r), q.e - P.off) + P.off - q.l)
    return late


# ----------------------------------------------------------------------------------------------------------------------
# Heurísticas
# ----------------------------------------------------------------------------------------------------------------------
def _legs(P, s, r, first_leg=True):
    """Distância em falta do pedido r (troços com carga; first_leg inclui a ida do veículo mais próximo ao produtor)."""
    D, C, c, q = P.D, P.C, s.status[r], P.req[r]
    if c == P.PENDING:
        x = D[q.p][C] + D[C][q.d]
        if first_leg:
            x += min(D[s.veh[v][0]][q.p] for v in P.allowed[r])
        return x
    if 1 <= c <= P.K:
        return D[s.veh[c - 1][0]][C] + D[C][q.d]
    if c == P.WH:
        return (min(D[s.veh[v][0]][C] for v in P.allowed[r]) if first_leg else 0.0) + D[C][q.d]
    if c == P.DONE:
        return 0.0
    return D[s.veh[c - P.ONB][0]][q.d]


def h_chain(P, s):
    """
    Heurística 1, cadeia mais longa (admissível). Cada pedido por terminar obriga a percorrer pelo menos os seus troços
    produtor, armazém, cliente, por isso a distância total nunca é inferior à do pedido mais exigente. Soma se o atraso
    inevitável (parte soft).
    """
    best = max((_legs(P, s, r) for r in range(P.n)), default=0.0)
    return best + P.late_penalty * late_bound(P, s)


def h_disjoint(P, s):
    """
    Heurística 2, janelas disjuntas. Dois pedidos cujas janelas [início da recolha, fim da entrega] não se sobrepõem nunca
    viajam juntos (se nenhum se atrasar), logo os troços "com carga" de cada um somam se. Escolhe se, por programação
    dinâmica (interval scheduling pesado), o conjunto de pedidos com janelas disjuntas que maximiza essa soma, e toma se o
    máximo com h1. Soma se o atraso inevitável. Com atrasos pode sobrestimar.
    """
    items = sorted((P.req[r].l, P.req[r].a, _legs(P, s, r, first_leg=False))
                   for r in range(P.n) if s.status[r] != P.DONE)
    best = [0.0] * (len(items) + 1)
    for i, (l, a, w) in enumerate(items, 1):
        j = i - 1
        while j > 0 and items[j - 1][0] >= a:
            j -= 1
        best[i] = max(best[i - 1], w + best[j])
    lb = late_bound(P, s)
    km = max(best[-1], h_chain(P, s) - P.late_penalty * lb)
    return km + P.late_penalty * lb


def h_work(P, s, lam=None):
    """
    Heurística 3, trabalho restante (NÃO admissível, só para o Greedy): soma das distâncias em falta de todos os pedidos
    + LAMBDA × etapas em falta (recolha, armazém, carga, entrega) + atraso inevitável.
    """
    lam = LAMBDA if lam is None else lam
    dist = steps = 0.0
    for r in range(P.n):
        c = s.status[r]
        if c == P.DONE:
            continue
        steps += 4 - P.stage(c)
        dist += _legs(P, s, r, first_leg=False)
    return dist + lam * steps + P.late_penalty * late_bound(P, s)


HEURISTICS = {"h1_cadeia": h_chain, "h2_disjuntas": h_disjoint, "h3_trabalho": h_work}


# ----------------------------------------------------------------------------------------------------------------------
# Pesquisa best first
# ----------------------------------------------------------------------------------------------------------------------
def plan_to_routes(P, plan):
    """Converte um plano [(veículo, ação), ...] em rotas: uma lista de ações por veículo."""
    routes = [[] for _ in range(P.K)]
    for v, act in plan:
        routes[v].append(act)
    return routes


def best_first(P, h, mode="astar", weight=1.0, max_nodes=200000, time_limit=60, use_dominance=True):
    """
    Pesquisa best first no espaço de estados.
      mode="greedy": f = h, com desempate pelo instante em que a ação acaba (a ação mais cedo primeiro);
      mode="astar" : f = g + weight*h (weight > 1 = A* ponderado).
    Cortes: estados mortos (janela de recolha, tempo no armazém) e dominância (mesmo estado físico com custo e tempos <=).
    Devolve dict(found, expanded, generated, seconds, cost, km, late_min, plan, timed_out).
    """
    t0, s0, cnt = time.time(), P.initial_state(), itertools.count()
    if mode == "greedy":
        prio = lambda g, hv, s, v: (hv, s.veh[v][1])
    else:
        prio = lambda g, hv, s, v: (g + weight * hv, 0.0)
    openh = [(prio(0.0, h(P, s0), s0, 0), next(cnt), 0.0, s0, False)]
    best_g, parent, seen = {s0: 0.0}, {s0: None}, {}
    expanded = generated = 0
    final, timed_out = None, False

    def dominated(s2, g2):
        pk = (tuple((x[0], x[3]) for x in s2.veh), s2.status, s2.src)
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
            timed_out = True
            break
        if P.is_goal(s):                                    # objetivo: acrescenta se o regresso ao centro como passo final
            g_end = g + P.return_cost(s)
            heapq.heappush(openh, (((-1.0, 0.0) if mode == "greedy" else (g_end, 0.0)), next(cnt), g_end, s, True))
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
                    heapq.heappush(openh, (prio(g2, h(P, s2), s2, v), next(cnt), g2, s2, False))
    out = dict(found=final is not None, expanded=expanded, generated=generated, seconds=time.time() - t0,
               cost=None, km=None, late_min=None, plan=None, timed_out=timed_out)
    if final:
        s, plan = final[1], []
        while parent[s] is not None:
            s, v, act = parent[s]; plan.append((v, act))
        plan.reverse()
        res = P.simulate(plan_to_routes(P, plan))            # custo recalculado pelo simulador do modelo
        out.update(plan=plan, cost=res["cost"], km=res["km"], late_min=res["late_min"])
    return out


def greedy(P, lam=None, **kw):
    """Greedy com a h3 (atalho para best_first(P, h_work, mode="greedy"))."""
    h = h_work if lam is None else (lambda P_, s: h_work(P_, s, lam))
    return best_first(P, h, mode="greedy", **kw)


# ----------------------------------------------------------------------------------------------------------------------
# Greedy construtivo cronológico
# ----------------------------------------------------------------------------------------------------------------------
def greedy_chrono(P, max_nodes=200000, time_limit=60, tie="cost"):
    """
    Greedy construtivo cronológico: a partir do estado atual escolhe sempre a ação viável que COMEÇA mais cedo (desempate
    pelo custo da ação, tie="cost"), como um despachante que trata primeiro o que acontece primeiro. Se chega a um estado
    sem saída (todas as ações cortadas por is_dead ou violam restrições hard) volta atrás e tenta a ação seguinte
    (backtracking limitado por max_nodes / time_limit). É uma pesquisa em profundidade guiada pela regra gulosa.
    Devolve o mesmo dicionário que best_first, mais o número de backtracks.
    """
    t0, s0 = time.time(), P.initial_state()

    def children(s):
        out = []
        for v in range(P.K):
            for act in P.actions(s, v):
                res = P.apply(s, v, act)
                if res is None or res is BLOCKED:
                    continue
                s2, km, late = res
                if is_dead(P, s2):
                    continue
                start = s2.veh[v][3] if s2.veh[v][3] is not None else s2.veh[v][1]
                out.append(((start, P.step_cost(km, late) if tie == "cost" else 0.0), v, act, s2))
        out.sort(key=lambda x: x[0])
        return out

    stack = [iter(children(s0))]          # um iterador de filhos por nível
    path, seen = [], {s0}
    expanded, backtracks, timed_out, found = 1, 0, False, False
    while stack:
        if P.is_goal(path[-1][2] if path else s0):
            found = True; break
        if expanded > max_nodes or time.time() - t0 > time_limit:
            timed_out = True; break
        nxt = next(stack[-1], None)
        if nxt is None:                   # sem mais filhos: volta atrás
            stack.pop()
            if path:
                path.pop(); backtracks += 1
            continue
        _, v, act, s2 = nxt
        if s2 in seen:
            continue
        seen.add(s2)
        path.append((v, act, s2))
        expanded += 1
        stack.append(iter(children(s2)))
    out = dict(found=found, expanded=expanded, generated=len(seen), backtracks=backtracks, seconds=time.time() - t0,
               cost=None, km=None, late_min=None, plan=None, timed_out=timed_out)
    if found:
        plan = [(v, act) for v, act, _ in path]
        res = P.simulate(plan_to_routes(P, plan))
        out.update(plan=plan, cost=res["cost"], km=res["km"], late_min=res["late_min"])
    return out
