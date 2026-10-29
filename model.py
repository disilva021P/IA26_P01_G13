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


# ----------------------------------------------------------------------------------------------------------------------
# Leitura dos dados
# ----------------------------------------------------------------------------------------------------------------------
def load_problem(data_dir="data", late_penalty=LATE_PENALTY):
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
