"""
model.py — Modelo do problema "Comida Local" (Trabalho de IA, UPCA 2026-27).

leitura dos dados, classes Request / Vehicle / State / Problem,
ações, função de custo e verificador de viabilidade de rotas. Os algoritmos (Greedy, A*, Hill Climbing, CSP)
importam daqui e usam sempre as mesmas regras.

Terminologia: uma ENCOMENDA (orders) é composta por um ou mais PEDIDOS (linhas de order_items).
O pedido é a unidade de planeamento: recolher no produtor -> passar pelo centro de recolha (armazém) -> entregar ao cliente.


Função de custo : custo = distância (km) + LATE_PENALTY * minutos de atraso.
"""
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path
import math

import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Parâmetros do modelo
# ----------------------------------------------------------------------------------------------------------------------
SERVICE_MIN = 10.0           # tempo de carga/descarga em cada paragem (min); 0 = instantâneo
LATE_PENALTY = 10.0        # penalização (km-equivalentes) por cada minuto de atraso face a delivery_end
RETURN_TO_CENTER = True     # os veículos terminam no centro de recolha a que pertencem
EPS = 1e-6

# Ações de um veículo
PICK = "P"      # P r : vai ao produtor do pedido r e carrega-o
LOAD = "L"      # L r : vai ao centro, descarrega o que trazia "cru" e carrega o pedido r (que já está no armazém)
CENTER = "C"    # C   : vai ao centro e descarrega no armazém tudo o que trazia "cru"
DELIVER = "D"   # D r : vai ao cliente e entrega o pedido r
BLOCKED = "blocked"   # a ação depende de outro veículo (o pedido ainda não chegou ao armazém)


# ----------------------------------------------------------------------------------------------------------------------
# Classes do modelo
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Request:
    """Pedido (linha de order_items). Tempos em minutos desde a origem (ver load_problem)."""
    id: int            # order_item_id
    order: int         # order_id da encomenda a que pertence
    product: str
    p: int             # location_id do produtor (recolha)
    d: int             # location_id do local de entrega
    w: float           # peso (kg) = quantity * unit_weight_kg
    a: float           # availability_start  (recolha não antes de)
    a_end: float       # availability_end    (recolha não depois de)
    e: float           # delivery_start      (entrega não antes de)
    l: float           # delivery_end        (entrega desejavelmente até; soft)


@dataclass(frozen=True)
class Vehicle:
    id: int
    cap: float         # capacidade (kg)
    center: int        # location_id do centro de recolha a que pertence


class State(NamedTuple):
    """
    Estado do problema (imutável e comparável, pode ser usado em conjuntos/dicionários):
      veh[v]    = (localização, tempo atual, carga em kg)   -> posição, tempo e carga transportada do veículo v
      status[r] = fase do pedido r (ver Problem): pendente | cru a bordo de v | no armazém | a bordo de v a caminho do cliente | entregue
      wt[r]     = instante em que o pedido r chegou ao armazém (0.0 se não estiver lá)
    """
    veh: tuple
    status: tuple
    wt: tuple


class Problem:
    """Instância do problema: pedidos, veículos, matrizes de distância (km) e tempo (min) indexadas por location_id."""

    def __init__(self, requests, vehicles, D, T, center, late_penalty=LATE_PENALTY):
        self.req, self.veh, self.D, self.T, self.C = list(requests), list(vehicles), D, T, center
        self.n, self.K = len(self.req), len(self.veh)
        self.w = [r.w for r in self.req]
        self.cap = [v.cap for v in self.veh]
        self.late_penalty = late_penalty
        # códigos da fase de cada pedido em State.status
        self.PENDING = 0                 # ainda no produtor
        # 1..K     -> recolhido pelo veículo v (código 1+v), a caminho do armazém ("cru")
        self.WH = self.K + 1             # no armazém
        self.ONB = self.K + 2            # a bordo do veículo v a caminho do cliente: código ONB+v
        self.DONE = 2 * self.K + 2       # entregue

    # ---------------------------------------------------------------- instâncias derivadas
    def subset(self, ids):
        """Sub-instância só com os pedidos indicados (por order_item_id)."""
        ids = set(ids)
        return Problem([r for r in self.req if r.id in ids], self.veh, self.D, self.T, self.C, self.late_penalty)

    def lateness_best_case(self, r):
        """Atraso mínimo (min) do pedido r face a delivery_end: recolha logo no início da janela, sem esperas (0 = pode cumprir)."""
        q = r if isinstance(r, Request) else self.req[r]
        t = q.a + SERVICE_MIN + self.T[q.p][self.C] + SERVICE_MIN + self.T[self.C][q.d]
        return max(0.0, max(t, q.e) - q.l)

    # ---------------------------------------------------------------- estado e objetivo
    def initial_state(self):
        """Todos os veículos no centro, tempo 0, vazios; todos os pedidos pendentes."""
        return State(tuple((v.center, 0.0, 0.0) for v in self.veh), tuple([self.PENDING] * self.n), tuple([0.0] * self.n))

    def is_goal(self, s):
        """Objetivo: todos os pedidos entregues."""
        return all(x == self.DONE for x in s.status)

    def return_cost(self, s):
        """Distância (km) para os veículos voltarem ao centro no fim."""
        if not RETURN_TO_CENTER:
            return 0.0
        return sum(self.D[loc][self.veh[v].center] for v, (loc, _, _) in enumerate(s.veh))

    def step_cost(self, km, late_min):
        """Função de custo de uma ação: distância + penalização do atraso."""
        return km + self.late_penalty * late_min

    # ---------------------------------------------------------------- ações
    def actions(self, s, v):
        """Ações possíveis (sintaticamente) do veículo v no estado s; apply() verifica as restrições."""
        out, carrying_raw = [], False
        for r in range(self.n):
            c = s.status[r]
            if c == self.PENDING:
                out.append((PICK, r))
            elif c == 1 + v:
                carrying_raw = True
                out.append((LOAD, r))
            elif c == self.WH:
                out.append((LOAD, r))
            elif c == self.ONB + v:
                out.append((DELIVER, r))
        if carrying_raw:
            out.append((CENTER, None))
        return out

    def apply(self, s, v, act):
        """
        Aplica a ação `act = (tipo, r)` ao veículo v.
        Devolve (novo_estado, km, atraso_min), ou None se viola uma restrição hard, ou BLOCKED se depende de outro veículo.
        O custo da ação é step_cost(km, atraso_min).
        """
        typ, r = act
        loc, t, load = s.veh[v]
        D, T, C = self.D, self.T, self.veh[v].center

        if typ == PICK:
            q = self.req[r]
            if s.status[r] != self.PENDING or load + q.w > self.cap[v] + EPS:
                return None
            arrive = t + T[loc][q.p]
            if arrive > q.a_end + EPS:                       # chega depois do fim da janela de recolha
                return None
            t2 = max(arrive, q.a) + SERVICE_MIN              # espera pelo início da janela
            return self._new(s, v, (q.p, t2, load + q.w), {r: (1 + v, 0.0)}), D[loc][q.p], 0.0

        if typ == DELIVER:
            q = self.req[r]
            if s.status[r] != self.ONB + v:
                return None
            t_del = max(t + T[loc][q.d], q.e)                # espera pelo início da janela de entrega
            late = max(0.0, t_del - q.l)                     # restrição soft: atraso penalizado
            return self._new(s, v, (q.d, t_del + SERVICE_MIN, load - q.w), {r: (self.DONE, 0.0)}), D[loc][q.d], late

        # LOAD / CENTER: o veículo vai ao centro, descarrega no armazém tudo o que trazia "cru" e (se LOAD) carrega r
        if typ == LOAD and s.status[r] not in (1 + v, self.WH):
            pend_or_other_raw = s.status[r] == self.PENDING or 1 <= s.status[r] <= self.K
            return BLOCKED if pend_or_other_raw else None
        t1 = t + T[loc][C]
        changes, load2 = {}, load
        for q in range(self.n):
            if s.status[q] == 1 + v:                         # descarrega os pedidos "crus" deste veículo
                changes[q] = (self.WH, round(t1, 4))
                load2 -= self.w[q]
        t2 = t1
        if typ == LOAD:
            wt_r = changes[r][1] if r in changes else s.wt[r]
            t2 = max(t1, wt_r)                               # espera, se preciso, que o pedido chegue ao armazém
            if load2 + self.w[r] > self.cap[v] + EPS:
                return None
            changes[r] = (self.ONB + v, 0.0)
            load2 += self.w[r]
        return self._new(s, v, (C, t2 + SERVICE_MIN, max(load2, 0.0)), changes), D[loc][C], 0.0

    @staticmethod
    def _new(s, v, veh_v, changes):
        veh = s.veh[:v] + ((veh_v[0], round(veh_v[1], 4), veh_v[2]),) + s.veh[v + 1:]
        status, wt = list(s.status), list(s.wt)
        for r, (code, w_time) in changes.items():
            status[r], wt[r] = code, w_time
        return State(veh, tuple(status), tuple(wt))

    # ---------------------------------------------------------------- rotas completas
    def simulate(self, routes, trace=False):
        """
        Executa as rotas (uma lista de ações por veículo) com as regras do modelo.
        Devolve dict(cost, km, late_min, late_by_request, log) ou None se alguma restrição hard for violada.
        As rotas dos vários veículos são executadas em paralelo: uma ação BLOCKED espera pela de outro veículo.
        """
        s, idx, km, late, late_req, log = self.initial_state(), [0] * self.K, 0.0, 0.0, {}, []
        progress = True
        while progress:
            progress = False
            for v in range(self.K):
                while idx[v] < len(routes[v]):
                    act = routes[v][idx[v]]
                    res = self.apply(s, v, act)
                    if res is None:
                        return None
                    if res is BLOCKED:
                        break
                    s2, d_km, d_late = res
                    km, late = km + d_km, late + d_late
                    if d_late > 0:
                        late_req[self.req[act[1]].id] = d_late
                    if trace:
                        log.append(dict(vehicle=self.veh[v].id, v=v, act=act[0], r=act[1], loc=s2.veh[v][0],
                                        t_end=s2.veh[v][1], km=d_km, late=d_late, load=s2.veh[v][2]))
                    s, idx[v], progress = s2, idx[v] + 1, True
        if any(idx[v] < len(routes[v]) for v in range(self.K)) or not self.is_goal(s):
            return None                                       # ações por executar (deadlock) ou pedidos por servir
        if RETURN_TO_CENTER:
            for v, (loc, t, _) in enumerate(s.veh):
                d_km = self.D[loc][self.veh[v].center]
                km += d_km
                if trace and d_km > 0:
                    log.append(dict(vehicle=self.veh[v].id, v=v, act="R", r=None, loc=self.veh[v].center,
                                    t_end=t + self.T[loc][self.veh[v].center], km=d_km, late=0.0, load=0.0))
        return dict(cost=self.step_cost(km, late), km=km, late_min=late, late_by_request=late_req, log=log)

    def cost(self, routes):
        """Função de custo de uma solução completa (inf se inviável)."""
        res = self.simulate(routes)
        return res["cost"] if res else math.inf

    def audit(self, routes):
        """
        Verificador de viabilidade, a partir do cronograma (não repete a lógica de apply):
        cada pedido recolhido e entregue 1 vez; recolha dentro da janela; entrega não antes de delivery_start;
        entrega fisicamente possível (produtor -> centro -> cliente); capacidade. Atrasos face a delivery_end são só reportados.
        """
        res = self.simulate(routes, trace=True)
        if res is None:
            return dict(ok=False, problemas=["a simulação falhou (restrição hard violada, pedido por servir ou deadlock)"])
        picks, delivs, load, max_load, problems = {}, {}, [0.0] * self.K, [0.0] * self.K, []
        for e in res["log"]:
            v, r = e["v"], e["r"]
            if e["act"] == PICK:
                if r in picks: problems.append(f"pedido {self.req[r].id} recolhido mais de uma vez")
                picks[r] = e["t_end"] - SERVICE_MIN
                load[v] += self.w[r]
            elif e["act"] == DELIVER:
                if r in delivs: problems.append(f"pedido {self.req[r].id} entregue mais de uma vez")
                delivs[r] = e["t_end"] - SERVICE_MIN
                load[v] -= self.w[r]
            max_load[v] = max(max_load[v], load[v])
        for r, q in enumerate(self.req):
            if r not in picks or r not in delivs:
                problems.append(f"pedido {q.id} não servido"); continue
            if not (q.a - EPS <= picks[r] <= q.a_end + EPS): problems.append(f"pedido {q.id}: recolha às {picks[r]:.0f} fora da janela")
            if delivs[r] < q.e - EPS: problems.append(f"pedido {q.id}: entregue antes de delivery_start")
            if delivs[r] < picks[r] + self.T[q.p][self.C] + self.T[self.C][q.d] - 1e-3:   # tolerância > arredondamento (4 casas) do estado
                problems.append(f"pedido {q.id}: entrega fisicamente impossível")
        for v in range(self.K):
            if max_load[v] > self.cap[v] + EPS: problems.append(f"capacidade do veículo {self.veh[v].id} excedida")
        return dict(ok=not problems, problemas=problems, km=res["km"], atraso_min=res["late_min"], custo=res["cost"],
                    pedidos_atrasados=res["late_by_request"], carga_max_kg=[round(x, 1) for x in max_load])

    def timeline(self, routes, origin=None):
        """Cronograma paragem a paragem como DataFrame (com data/hora se `origin` for dado)."""
        res = self.simulate(routes, trace=True)
        rows = [dict(veiculo=e["vehicle"], acao=e["act"], pedido=(self.req[e["r"]].id if e["r"] is not None else None),
                     local=e["loc"], t_fim=e["t_end"], km=round(e["km"], 2), atraso_min=round(e["late"], 1)) for e in res["log"]]
        out = pd.DataFrame(rows)
        if origin is not None:
            out["hora"] = [(origin + pd.Timedelta(minutes=t)).strftime("%d/%m %H:%M") for t in out.t_fim]
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Leitura dos dados
# ----------------------------------------------------------------------------------------------------------------------
def load_problem(data_dir="data_tiny", late_penalty=LATE_PENALTY):
    """
    Lê os CSV de `data_dir` e devolve (Problem, origin).
    Os instantes ficam em minutos desde `origin` = 1 dia antes da primeira janela de recolha (os veículos começam no tempo 0).
    """
    rd = lambda f, **k: pd.read_csv(Path(data_dir) / f, **k)
    loc, prod, vehs, orders, dm = rd("locations.csv"), rd("products.csv"), rd("vehicles.csv"), rd("orders.csv"), rd("distance_matrix.csv")
    items = rd("order_items.csv", parse_dates=["availability_start", "availability_end", "delivery_start", "delivery_end"])
    items = items.merge(prod, on="product_id", suffixes=("", "_prod")).merge(orders, on="order_id")

    p_loc = loc[loc.location_type == "producer"].set_index("ref_id").location_id
    d_loc = loc[loc.location_type == "delivery_location"].set_index("ref_id").location_id
    c_loc = loc[loc.location_type == "collection_center"].set_index("ref_id").location_id   # center_id -> location_id
    center = int(c_loc.iloc[0])

    origin = items.availability_start.min() - pd.Timedelta(days=1)
    mins = lambda col: (col - origin).dt.total_seconds() / 60
    items["a"], items["a_end"], items["e"], items["l"] = (mins(items[c]) for c in
        ("availability_start", "availability_end", "delivery_start", "delivery_end"))

    m = int(loc.location_id.max()) + 1
    D, T = [[0.0] * m for _ in range(m)], [[0.0] * m for _ in range(m)]
    for r in dm.itertuples():
        D[r.origin_location_id][r.destination_location_id] = r.distance_km
        T[r.origin_location_id][r.destination_location_id] = r.duration_min

    reqs = [Request(int(r.order_item_id), int(r.order_id), r.name, int(p_loc[r.producer_id]), int(d_loc[r.delivery_location_id]),
                    float(r.quantity * r.unit_weight_kg), r.a, r.a_end, r.e, r.l) for r in items.itertuples()]
    vs = [Vehicle(int(v.vehicle_id), float(v.capacity_kg), int(c_loc[v.center_id])) for v in vehs.itertuples()]
    return Problem(reqs, vs, D, T, center, late_penalty), origin


# ----------------------------------------------------------------------------------------------------------------------
# Teste rápido:  python model.py
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    P, origin = load_problem()
    print(f"{P.n} pedidos, {P.K} veículos, centro = location {P.C}, carga total {sum(P.w):.1f} kg")
    late = {r.id: round(P.lateness_best_case(r), 1) for r in P.req if P.lateness_best_case(r) > 0}
    print("Pedidos que não cumprem delivery_end nem no melhor caso (min de atraso):", late)

    # Solução ingénua (só para testar o modelo): por ordem de início da recolha, cada pedido (P, L, D) vai para o primeiro veículo
    # em que a rota continua viável. As rotas guardam order_item_id; conv() converte para os índices de cada sub-instância.
    conv = lambda Q, rts: [[(a, None if i is None else [q.id for q in Q.req].index(i)) for a, i in rt] for rt in rts]
    ids, routes = [], [[] for _ in range(P.K)]
    for r in sorted(range(P.n), key=lambda r: P.req[r].a):
        rid = P.req[r].id
        for v in range(P.K):
            trial = [list(rt) for rt in routes]
            trial[v] += [(PICK, rid), (LOAD, rid), (DELIVER, rid)]
            if P.subset(ids + [rid]).simulate(conv(P.subset(ids + [rid]), trial)):
                routes, ids = trial, ids + [rid]
                break
        else:
            print(f"  pedido {rid} não cabe em nenhum veículo nesta heurística ingénua (fica por servir)")
    Q = P.subset(ids)
    routes = conv(Q, routes)
    print(f"Rota ingénua: {len(ids)} de {P.n} pedidos servidos")
    res = Q.simulate(routes)
    print("Custo:", {k: round(v, 1) for k, v in res.items() if k in ("cost", "km", "late_min")})
    print("Auditoria:", Q.audit(routes))
