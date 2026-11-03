"""
csp.py: o problema formulado como CSP (Constraint Satisfaction Problem) sobre o modelo de model.py.

Formulação: 4 variáveis por pedido, uma por PARAGEM (evento), e não uma por viagem:
    P_r  recolha no produtor | C_r  descarga no armazém pelo veículo que recolheu |
    L_r  carga no armazém pelo veículo que vai entregar | D_r  entrega no cliente
Domínio de cada variável = (veículo, instante de início do serviço), com o tempo discretizado em SLOT minutos.
O veículo que recolhe (P_r, C_r) pode ser diferente do que entrega (L_r, D_r): é a passagem pelo armazém do modelo.
Quando é o mesmo, a descarga é feita pela própria L_r e C_r toma o valor "virtual" (veículo, None): não é paragem.
Durações: S nas paragens P e D, H nas paragens C e L do armazém. O mesmo veículo no mesmo local à mesma hora conta como
uma só paragem (vários artigos no mesmo serviço).
As viagens com vários pedidos não são impostas: aparecem quando paragens de pedidos diferentes ficam seguidas no mesmo
veículo, porque o custo é a distância entre paragens consecutivas.
"""
import math, random, time
from dataclasses import dataclass

import numpy as np

from model import PICK, LOAD, CENTER, DELIVER, EPS

SLOT = 5           # granularidade temporal das variáveis (min); o custo final é sempre recalculado por P.simulate
MAX_LATE = 0       # folga de atraso (min) permitida nos domínios, além do atraso inevitável de cada pedido (soft)
KINDS = (PICK, CENTER, LOAD, DELIVER)              # ordem das 4 variáveis de cada pedido: índice da variável = 4 * r + posição
KIND_RANK = {k: i for i, k in enumerate(KINDS)}


@dataclass
class Job:
    """Variável do CSP: paragem kind (P, C, L ou D) do pedido r, no local loc. Domínio: lista de pares (veículo, hora)."""
    idx: int; r: int; kind: str; loc: int; domain: list; due: float = None


def _grid(lo, hi, slot):
    return [float(t) for t in np.arange(math.ceil(lo / slot - EPS) * slot, hi + EPS, slot)]


def duration(P, kind):
    """Duração do serviço de uma paragem: S na recolha e na entrega, H no armazém."""
    return P.S if kind in (PICK, DELIVER) else P.H


def build_jobs(P, max_late=MAX_LATE, slot=SLOT):
    """
    4 variáveis por pedido. Os limites dos domínios vêm das restrições unárias e da propagação das precedências
    P, C, L, D (tempos mínimos de viagem e de serviço), por isso já nascem podados.
      P: início em [availability_start, availability_end] (desviado de S se as janelas forem sobre o fim do serviço)
      D: início em [delivery_start, delivery_end + atraso inevitável + max_late]  (início hard; fim soft, com penalização)
      C, L: o que sobra entre os dois. C tem ainda o valor virtual (veículo, None).
      Veículos: só os que têm capacidade para o pedido e, se o pedido exigir frio, refrigerados.
    """
    S, H, T, C, off, jobs = P.S, P.H, P.T, P.C, P.off, []
    for r, q in enumerate(P.req):
        vs = [v for v in P.allowed[r] if P.w[r] <= P.cap[v] + EPS]                # restrições unárias: frio e capacidade
        d_pl = S + T[q.p][C]                                                       # início de P até chegar ao armazém
        d_ld = H + T[C][q.d]                                                       # início de L até chegar ao cliente
        p_lo = q.a - off
        l_lo = p_lo + d_pl
        d_lo = max(q.e - off, l_lo + d_ld)
        d_hi = q.l - off + math.ceil(P.lateness_best_case(q) / slot) * slot + max_late + slot
        l_hi = d_hi - d_ld
        p_hi = min(q.a_end - off, l_hi - d_pl)
        real = lambda lo, hi: [(v, t) for t in _grid(lo, hi, slot) for v in vs]
        doms = (real(p_lo, p_hi), [(v, None) for v in vs] + real(l_lo, l_hi - H), real(l_lo, l_hi), real(d_lo, d_hi))
        for kind, loc, dom in zip(KINDS, (q.p, C, C, q.d), doms):
            jobs.append(Job(len(jobs), r, kind, loc, dom, q.l - off if kind == DELIVER else None))
    return jobs


def csp_compatible(P, ji, vi, ti, jj, vj, tj):
    """
    Restrições binárias (ti/tj = None só na descarga virtual).
      mesmo pedido : P e C no mesmo veículo, C depois da viagem; C virtual: L no mesmo veículo da recolha;
                     C real: L só depois do fim da descarga (C + H) e no máximo max_warehouse_dwell depois;
                     L e D no mesmo veículo, D depois de L + H + viagem; se o mesmo veículo faz P e L (ou P e D),
                     respeita se o tempo de serviço e de viagem entre eles;
      mesmo veículo: duas paragens não se sobrepõem (serviço + viagem entre os locais), em qualquer ordem, exceto se
                     forem no mesmo local à mesma hora (uma só paragem com vários artigos).
    """
    S, H, T, C = P.S, P.H, P.T, P.C
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
            if vi == vj and tj < ti + S + T[ji.loc][C] + H + T[C][jj.loc] - EPS:
                return False
        elif pair == (CENTER, LOAD):
            if ti is None:
                if vi != vj:
                    return False
            elif tj < ti + H - EPS or tj - (ti + H) > P.dwell + EPS:
                return False
        elif pair == (LOAD, DELIVER):
            if vi != vj or tj < ti + H + T[C][jj.loc] - EPS:
                return False
        elif pair == (CENTER, DELIVER):
            if ti is not None and tj < ti + 2 * H + T[C][jj.loc] - EPS:
                return False
    if ti is None or tj is None or vi != vj:
        return True
    if ji.loc == jj.loc and abs(ti - tj) < EPS and (ji.kind in (PICK, DELIVER)) == (jj.kind in (PICK, DELIVER)):
        return True                                       # mesma paragem (mesmo veículo, local e hora)
    di, dj = duration(P, ji.kind), duration(P, jj.kind)
    return ti + di + T[ji.loc][jj.loc] <= tj + EPS or tj + dj + T[jj.loc][ji.loc] <= ti + EPS


def csp_neighbors(P, jobs):
    """Só ligam variáveis cujos intervalos possam colidir (paragens muito afastadas no tempo são independentes)."""
    span = []
    for j in jobs:
        ts = [t for _, t in j.domain if t is not None]
        span.append((min(ts), max(ts) + max(P.S, P.H)) if ts else (0.0, 0.0))
    mt = max(max(row) for row in P.T) + max(P.S, P.H)
    nb = {j.idx: [] for j in jobs}
    for i in jobs:
        for j in jobs:
            if i.idx < j.idx and (i.r == j.r or (span[i.idx][0] < span[j.idx][1] + mt and span[j.idx][0] < span[i.idx][1] + mt)):
                nb[i.idx].append(j.idx); nb[j.idx].append(i.idx)
    return nb


def capacity_ok(P, jobs, assigned, v):
    """
    Restrição global de capacidade do veículo v. Um pedido está a bordo do veículo que recolhe desde P até à descarga
    (C real, ou a entrega se a descarga é virtual) e a bordo do que entrega desde L até D (limite conservador: ignora o
    tempo no armazém). Só conta intervalos já conhecidos, por isso nunca poda uma atribuição que ainda possa ser completada.
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
    Custo planeado (km + penalização de atrasos) de uma atribuição parcial ou completa: por veículo, centro, paragens por
    ordem temporal, centro. Inserir uma paragem nunca reduz a distância (desigualdade triangular), logo o custo parcial é
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
    """
    Ordena as paragens de cada veículo pelo instante e traduz para ações do modelo (P r, C, L r, D r). Uma descarga C só
    é emitida se o veículo ainda traz produtos "crus" (uma carga L no armazém já descarrega tudo).
    """
    routes = [[] for _ in range(P.K)]
    for v in range(P.K):
        raw = False
        for _, _, _, i in _ordered(jobs, assigned, v):
            kind = jobs[i].kind
            if kind == CENTER:
                if raw:
                    routes[v].append((CENTER, None)); raw = False
                continue
            routes[v].append((kind, jobs[i].r))
            if kind == PICK:
                raw = True
            elif kind == LOAD:
                raw = False
    return routes


class _Restart(Exception):
    pass


def csp_solve(P, var_order="static", value_order="lex", forward_checking=False, max_nodes=300000, time_limit=60,
              optimize=False, max_late=MAX_LATE, slot=SLOT, restarts=0, restart_nodes=2000, seed=0):
    """
    Backtracking. var_order: "static" | "mrv" (menos valores legais, desempate por grau). value_order: "lex" | "soft"
    (menor custo extra). forward_checking: filtra os domínios dos vizinhos a cada atribuição. optimize=True: depois da
    1ª solução continua em branch & bound (poda pelo custo parcial) até max_nodes/time_limit. Conta nós e backtracks.
    Devolve também timed_out e complete (True só se o branch & bound esgotou a árvore: o custo planeado é então ótimo
    para esta formulação). O custo final (cost) é recalculado por P.simulate, que executa cada rota o mais cedo possível.
    restarts > 0: reinícios aleatórios (contra o thrashing do backtracking cronológico). Cada tentativa tem um orçamento
    de restart_nodes nós (que cresce 1.5x a cada reinício); a partir da 2ª, os desempates e os valores são perturbados.
    """
    jobs, t0 = build_jobs(P, max_late, slot), time.time()
    out = dict(found=False, first=None, nodes=0, backtracks=0, seconds=0.0, timed_out=False, complete=False,
               cost=None, km=None, late_min=None, cost_planned=None, routes=None, assigned=None, jobs=jobs)
    if any(not j.domain for j in jobs):                    # domínio vazio logo à partida: pedido impossível com este max_late
        return out
    nb = csp_neighbors(P, jobs)
    dom = {j.idx: list(j.domain) for j in jobs}
    assigned, st = {}, dict(nodes=0, backtracks=0)
    best = dict(cost=math.inf, assigned=None, first=None)
    tkey = lambda x: -1.0 if x[1] is None else x[1]
    rng, attempt = random.Random(seed), dict(k=0, budget=math.inf, nodes0=0)
    JITTER = 30.0                                           # perturbação máxima (km equivalentes) dos valores nos reinícios

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
        out["assigned"] = best["assigned"]
        out["routes"] = csp_to_routes(P, jobs, best["assigned"])
        out["cost_planned"] = best["cost"]
        res = P.simulate(out["routes"])
        if res:
            out.update(cost=res["cost"], km=res["km"], late_min=res["late_min"])
    return out


def csp_complete(P, limits=(0, 15, 30, 60, 120, 480, 1440), **kw):
    """
    Relaxamento gradual da restrição soft: começa por só permitir o atraso inevitável de cada pedido e alarga a folga de
    atraso dos domínios (limits, em min) até haver solução. Com a penalização alta, as boas soluções quase não têm atraso
    evitável. O tempo limite (time_limit) é partilhado por todas as tentativas.
    """
    kw.setdefault("restarts", 12)
    budget, t0, out = kw.pop("time_limit", 60), time.time(), None
    for L in limits:
        left = budget - (time.time() - t0)
        if left <= 0:
            break
        out = csp_solve(P, max_late=L, time_limit=left, **kw)
        if out["found"]:
            out["max_late_used"] = L
            out["seconds"] = time.time() - t0
            return out
    if out is not None:
        out["seconds"] = time.time() - t0
    return out


def assignment_table(P, res, origin=None):
    """Atribuição viável do CSP como tabela: variável, pedido, veículo e hora de início (None = descarga virtual)."""
    import pandas as pd
    rows = []
    for i, (v, t) in sorted(res["assigned"].items()):
        j = res["jobs"][i]
        hora = None if t is None else ((origin + pd.Timedelta(minutes=t)).strftime("%d/%m %H:%M") if origin is not None else t)
        rows.append(dict(variavel=f"{j.kind}_{P.req[j.r].id}", pedido=P.req[j.r].id, veiculo=P.veh[v].id, inicio=hora))
    return pd.DataFrame(rows)
