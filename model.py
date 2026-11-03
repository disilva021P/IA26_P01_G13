"""
model.py: modelo do problema "Comida Local" (Trabalho de IA, UPCA 2026-27, grupo G13).

Leitura dos dados e das regras (rules.csv), classes Rules / Request / Vehicle / State / Problem, ações, função de custo,
simulador de rotas e verificador de viabilidade. Os algoritmos (informed.py, local_search.py, csp.py, agent.py) importam
daqui e usam sempre as mesmas regras.

Terminologia: uma ENCOMENDA (orders) é composta por um ou mais PEDIDOS (linhas de order_items).
O pedido é a unidade de planeamento: recolher no produtor, passar pelo centro de recolha (armazém) e entregar ao cliente.

Regras (lidas de rules.csv; sem o ficheiro usam se os valores por omissão de Rules)
  stop_service_min        S : duração do serviço numa PARAGEM de recolha ou de entrega. Uma paragem são ações seguidas do
                              mesmo veículo no mesmo local (vários artigos no mesmo produtor ou entregues no mesmo local
                              pagam um só serviço). A ação junta se à paragem em curso se o produto já estava pronto no
                              início dessa paragem; se não, começa um novo serviço.
  warehouse_handling_min  H : duração de uma paragem no armazém, para descarregar e/ou carregar (INTERPRETAÇÃO: uma só
                              operação de H minutos cobre a descarga e a carga feitas na mesma paragem).
  max_warehouse_dwell_h     : tempo máximo (hard) entre a chegada ao armazém (fim da descarga) e o início da carga.
  window_semantics          : "service_start" (as janelas limitam o início do serviço) ou "service_end" (o fim).
  flow                      : só se suporta producer>warehouse>delivery_location.
  start_date                : a origem do tempo é a meia noite deste dia (os veículos começam no tempo 0).
  avg_speed_kmh, route_factor: só informativos (a matriz de distâncias e tempos já vem construída com eles).

Restrições
  hard : recolha com início (ou fim) dentro de [availability_start, availability_end]; entrega não antes de
         delivery_start; capacidade do veículo; produtos que exigem frio só em veículos refrigerados; cada pedido
         recolhido e entregue uma vez; todo o produto passa pelo armazém; só se carrega no armazém o que já lá chegou;
         tempo máximo no armazém.
  soft : entrega até delivery_end, com penalização LATE_PENALTY por minuto de atraso.

Função de custo: custo = distância (km) + LATE_PENALTY * minutos de atraso.
"""
from dataclasses import dataclass, field
from typing import NamedTuple
from pathlib import Path
import math
import sys

import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Parâmetros gerais
# ----------------------------------------------------------------------------------------------------------------------
LATE_PENALTY = 10.0         # penalização (km equivalentes) por cada minuto de atraso face a delivery_end
RETURN_TO_CENTER = True     # os veículos terminam no centro de recolha a que pertencem
EPS = 1e-6                  # tolerância nas comparações entre números decimais

# Ações de um veículo
PICK = "P"      # P r : vai ao produtor do pedido r e carrega o
LOAD = "L"      # L r : vai ao armazém, descarrega o que trazia "cru" e carrega o pedido r (que já lá está, ou que trazia)
CENTER = "C"    # C   : vai ao armazém e descarrega tudo o que trazia "cru"
DELIVER = "D"   # D r : vai ao cliente e entrega o pedido r
BLOCKED = "blocked"   # a ação depende de outro veículo (o pedido ainda não chegou ao armazém)

FLOW_SUPPORTED = "producer>warehouse>delivery_location"
COLD_COLUMNS = ("requires_refrigeration", "refrigerated", "requires_cold", "cold", "needs_refrigeration")


# ----------------------------------------------------------------------------------------------------------------------
# Regras do dataset
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Rules:
    """Regras do dataset. Os valores por omissão reproduzem o modelo antigo (sem rules.csv)."""
    start_date: str = None                   # sem regras: origem = 1 dia antes da primeira recolha
    horizon_days: int = None
    flow: str = FLOW_SUPPORTED
    stop_service_min: float = 10.0           # S
    warehouse_handling_min: float = 10.0     # H (no modelo antigo o armazém também custava 10 min)
    max_warehouse_dwell_h: float = math.inf  # sem limite
    window_semantics: str = "service_start"
    avg_speed_kmh: float = None
    route_factor: float = None
    generator_seed: int = None
    source: str = "valores por omissão"
    extra: dict = field(default_factory=dict)   # regras desconhecidas (guardadas, não usadas)

    def table(self):
        """Regras como DataFrame (para mostrar nos notebooks)."""
        rows = [(k, getattr(self, k)) for k in ("start_date", "horizon_days", "flow", "stop_service_min",
                "warehouse_handling_min", "max_warehouse_dwell_h", "window_semantics", "avg_speed_kmh",
                "route_factor", "generator_seed")]
        rows += [(k, v) for k, v in self.extra.items()]
        return pd.DataFrame(rows, columns=["regra", "valor"])


def load_rules(data_dir):
    """Lê data_dir/rules.csv (colunas rule, value). Se não existir, devolve os valores por omissão."""
    path = Path(data_dir) / "rules.csv"
    r = Rules()
    if not path.exists():
        return r
    conv = {"start_date": str, "horizon_days": int, "flow": str, "stop_service_min": float,
            "warehouse_handling_min": float, "max_warehouse_dwell_h": float, "window_semantics": str,
            "avg_speed_kmh": float, "route_factor": float, "generator_seed": int}
    for row in pd.read_csv(path).itertuples():
        key, val = str(row.rule).strip(), str(row.value).strip()
        if key in conv:
            setattr(r, key, conv[key](val))
        else:
            r.extra[key] = val
    r.source = str(path)
    return r


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
    w: float           # peso (kg)
    a: float           # availability_start  (recolha não antes de)
    a_end: float       # availability_end    (recolha não depois de)
    e: float           # delivery_start      (entrega não antes de)
    l: float           # delivery_end        (entrega desejavelmente até; soft)
    cold: bool = False # exige veículo refrigerado


@dataclass(frozen=True)
class Vehicle:
    id: int
    cap: float                 # capacidade (kg)
    center: int                # location_id do centro de recolha a que pertence
    refrigerated: bool = False


class State(NamedTuple):
    """
    Estado do problema (imutável e comparável, pode ser usado em conjuntos e dicionários):
      veh[v]    = (localização, tempo livre, carga em kg, início do serviço da paragem em curso ou None)
      status[r] = fase do pedido r: pendente | cru a bordo de v | no armazém | a bordo de v a caminho do cliente | entregue
      wt[r]     = instante em que o pedido r chegou ao armazém (fim da descarga; 0.0 se não estiver lá)
      src[r]    = veículo que descarregou r no armazém (-1 se não estiver lá)
    """
    veh: tuple
    status: tuple
    wt: tuple
    src: tuple


class Problem:
    """Instância do problema: pedidos, veículos, matrizes de distância (km) e tempo (min) indexadas por location_id."""

    def __init__(self, requests, vehicles, D, T, center, late_penalty=LATE_PENALTY, rules=None):
        self.req, self.veh, self.D, self.T, self.C = list(requests), list(vehicles), D, T, center
        self.n, self.K = len(self.req), len(self.veh)
        self.w = [r.w for r in self.req]
        self.cap = [v.cap for v in self.veh]
        self.late_penalty = late_penalty
        self.rules = rules or Rules()
        if self.rules.flow != FLOW_SUPPORTED:
            raise NotImplementedError(f"flow '{self.rules.flow}' não suportado (só {FLOW_SUPPORTED})")
        if self.rules.window_semantics not in ("service_start", "service_end"):
            raise NotImplementedError(f"window_semantics '{self.rules.window_semantics}' não suportado")
        self.S = float(self.rules.stop_service_min)                 # serviço numa paragem de recolha/entrega
        self.H = float(self.rules.warehouse_handling_min)           # paragem no armazém
        self.dwell = float(self.rules.max_warehouse_dwell_h) * 60   # min (inf = sem limite)
        self.off = self.S if self.rules.window_semantics == "service_end" else 0.0   # desvio das janelas
        # veículos que podem levar cada pedido (frio)
        self.allowed = [tuple(v for v in range(self.K) if (self.veh[v].refrigerated or not q.cold)) for q in self.req]
        # códigos da fase de cada pedido em State.status
        self.PENDING = 0                 # ainda no produtor
        # 1..K     : recolhido pelo veículo v (código 1+v), a caminho do armazém ("cru")
        self.WH = self.K + 1             # no armazém
        self.ONB = self.K + 2            # a bordo do veículo v a caminho do cliente: código ONB+v
        self.DONE = 2 * self.K + 2       # entregue

    # ---------------------------------------------------------------- instâncias derivadas
    def subset(self, ids):
        """Sub instância só com os pedidos indicados (por order_item_id)."""
        ids = set(ids)
        return Problem([r for r in self.req if r.id in ids], self.veh, self.D, self.T, self.C, self.late_penalty, self.rules)

    def stage(self, c):
        """Etapa (0 a 4) de um código de status: pendente, cru, armazém, a bordo, entregue."""
        if c == self.PENDING: return 0
        if c <= self.K: return 1
        if c == self.WH: return 2
        if c == self.DONE: return 4
        return 3

    def lateness_best_case(self, r):
        """Atraso mínimo (min) do pedido r: recolha logo que possível, a + S + T(p,C) + H + T(C,d), sem esperas."""
        q = r if isinstance(r, Request) else self.req[r]
        t = (q.a - self.off) + self.S + self.T[q.p][self.C] + self.H + self.T[self.C][q.d]
        return max(0.0, max(t, q.e - self.off) + self.off - q.l)

    # ---------------------------------------------------------------- estado e objetivo
    def initial_state(self):
        """Todos os veículos no centro, tempo 0, vazios; todos os pedidos pendentes."""
        n = self.n
        return State(tuple((v.center, 0.0, 0.0, None) for v in self.veh), (self.PENDING,) * n, (0.0,) * n, (-1,) * n)

    def is_goal(self, s):
        """Objetivo: todos os pedidos entregues."""
        return all(x == self.DONE for x in s.status)

    def return_cost(self, s):
        """Distância (km) para os veículos voltarem ao centro no fim."""
        if not RETURN_TO_CENTER:
            return 0.0
        return sum(self.D[x[0]][self.veh[v].center] for v, x in enumerate(s.veh))

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
                if v in self.allowed[r]:
                    out.append((PICK, r))
            elif c == 1 + v:
                carrying_raw = True
                out.append((LOAD, r))
            elif c == self.WH:
                if v in self.allowed[r]:
                    out.append((LOAD, r))
            elif c == self.ONB + v:
                out.append((DELIVER, r))
        if carrying_raw:
            out.append((CENTER, None))
        return out

    def apply(self, s, v, act):
        """
        Aplica a ação act = (tipo, r) ao veículo v.
        Devolve (novo_estado, km, atraso_min), ou None se viola uma restrição hard, ou BLOCKED se depende de outro veículo.
        O custo da ação é step_cost(km, atraso_min).
        """
        typ, r = act
        loc, t, load, ss = s.veh[v]
        D, T, S, H, off = self.D, self.T, self.S, self.H, self.off
        C = self.veh[v].center

        if typ == PICK:
            q = self.req[r]
            if s.status[r] != self.PENDING or v not in self.allowed[r] or load + q.w > self.cap[v] + EPS:
                return None
            if loc == q.p and ss is not None and q.a - off <= ss + EPS and ss <= q.a_end - off + EPS:
                # junta se à paragem em curso no mesmo produtor (o produto já estava pronto no início do serviço)
                return self._new(s, v, (loc, t, load + q.w, ss), {r: (1 + v, 0.0, -1)}), 0.0, 0.0
            start = max(t + T[loc][q.p], q.a - off)               # espera pelo início da janela
            if start > q.a_end - off + EPS:                       # já não consegue recolher dentro da janela
                return None
            return self._new(s, v, (q.p, start + S, load + q.w, start), {r: (1 + v, 0.0, -1)}), D[loc][q.p], 0.0

        if typ == DELIVER:
            q = self.req[r]
            if s.status[r] != self.ONB + v:
                return None
            if loc == q.d and ss is not None and q.e - off <= ss + EPS:
                start, t2, km = ss, t, 0.0                        # junta se à entrega em curso no mesmo local
            else:
                start = max(t + T[loc][q.d], q.e - off)           # espera pelo início da janela de entrega
                t2, km = start + S, D[loc][q.d]
            late = max(0.0, start + off - q.l)                    # restrição soft: atraso penalizado
            return self._new(s, v, (q.d, t2, load - q.w, start), {r: (self.DONE, 0.0, -1)}), km, late

        # LOAD / CENTER no armazém
        if typ == LOAD:
            q = self.req[r]
            c = s.status[r]
            if c not in (1 + v, self.WH):
                return BLOCKED if (c == self.PENDING or 1 <= c <= self.K) else None
            if v not in self.allowed[r]:
                return None
            if c == self.WH and loc == C and ss is not None and (
                    s.wt[r] <= ss + EPS or (s.src[r] == v and abs(s.wt[r] - (ss + H)) < 1e-3)):
                # junta se à paragem em curso no armazém: o produto já lá estava, ou foi descarregado por este veículo
                if ss - s.wt[r] > self.dwell + EPS or load + q.w > self.cap[v] + EPS:
                    return None
                return self._new(s, v, (loc, t, load + q.w, ss), {r: (self.ONB + v, 0.0, -1)}), 0.0, 0.0
        elif not any(c == 1 + v for c in s.status):
            return None                                           # C sem nada para descarregar

        # nova paragem no armazém: descarrega tudo o que trazia "cru" e (se LOAD) carrega r
        t1 = t + T[loc][C]
        start = t1
        if typ == LOAD and s.status[r] == self.WH:
            start = max(t1, s.wt[r])                              # espera que o produto chegue ao armazém
            if start - s.wt[r] > self.dwell + EPS:                # excedeu o tempo máximo no armazém
                return None
        end = start + H
        changes, load2 = {}, load
        for k in range(self.n):
            if s.status[k] == 1 + v:
                if typ == LOAD and k == r:
                    changes[k] = (self.ONB + v, 0.0, -1)          # o próprio pedido fica a bordo
                else:
                    changes[k] = (self.WH, round(end, 4), v)      # chega ao armazém no fim da descarga
                    load2 -= self.w[k]
        if typ == LOAD and s.status[r] == self.WH:
            if load2 + self.w[r] > self.cap[v] + EPS:
                return None
            changes[r] = (self.ONB + v, 0.0, -1)
            load2 += self.w[r]
        return self._new(s, v, (C, end, max(load2, 0.0), start), changes), D[loc][C], 0.0

    @staticmethod
    def _new(s, v, veh_v, changes):
        x = (veh_v[0], round(veh_v[1], 4), round(veh_v[2], 6), None if veh_v[3] is None else round(veh_v[3], 4))
        veh = s.veh[:v] + (x,) + s.veh[v + 1:]
        if not changes:
            return State(veh, s.status, s.wt, s.src)
        status, wt, src = list(s.status), list(s.wt), list(s.src)
        for r, (code, w_time, sv) in changes.items():
            status[r], wt[r], src[r] = code, w_time, sv
        return State(veh, tuple(status), tuple(wt), tuple(src))

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
                        unloaded = [k for k in range(self.n) if s.status[k] == 1 + v and s2.status[k] == self.WH]
                        log.append(dict(vehicle=self.veh[v].id, v=v, act=act[0], r=act[1], loc=s2.veh[v][0],
                                        t_start=s2.veh[v][3], t_end=s2.veh[v][1], km=d_km, late=d_late,
                                        load=s2.veh[v][2], unloaded=unloaded))
                    s, idx[v], progress = s2, idx[v] + 1, True
        if any(idx[v] < len(routes[v]) for v in range(self.K)) or not self.is_goal(s):
            return None                                       # ações por executar (deadlock) ou pedidos por servir
        if RETURN_TO_CENTER:
            for v, (loc, t, _, _) in enumerate(s.veh):
                d_km = self.D[loc][self.veh[v].center]
                km += d_km
                if trace and d_km > 0:
                    t_ret = t + self.T[loc][self.veh[v].center]
                    log.append(dict(vehicle=self.veh[v].id, v=v, act="R", r=None, loc=self.veh[v].center,
                                    t_start=t_ret, t_end=t_ret, km=d_km, late=0.0, load=0.0, unloaded=[]))
        return dict(cost=self.step_cost(km, late), km=km, late_min=late, late_by_request=late_req, log=log)

    def cost(self, routes):
        """Função de custo de uma solução completa (inf se inviável)."""
        res = self.simulate(routes)
        return res["cost"] if res else math.inf

    def audit(self, routes):
        """
        Verificador de viabilidade a partir do cronograma (não repete a lógica de apply): cada pedido recolhido e entregue
        uma vez; recolha dentro da janela; entrega não antes de delivery_start; entrega fisicamente possível
        (produtor, armazém, cliente); capacidade; frio; tempo máximo no armazém. Os atrasos face a delivery_end são só
        reportados.
        """
        res = self.simulate(routes, trace=True)
        if res is None:
            return dict(ok=False, problemas=["a simulação falhou (restrição hard violada, pedido por servir ou deadlock)"])
        picks, delivs, arrived, loaded = {}, {}, {}, {}
        load, max_load, problems = [0.0] * self.K, [0.0] * self.K, []
        for e in res["log"]:
            v, r = e["v"], e["r"]
            for k in e["unloaded"]:
                arrived[k] = e["t_end"]
                load[v] -= self.w[k]
            if e["act"] == PICK:
                if r in picks: problems.append(f"pedido {self.req[r].id} recolhido mais de uma vez")
                picks[r] = e["t_start"]
                load[v] += self.w[r]
                if self.req[r].cold and not self.veh[v].refrigerated:
                    problems.append(f"pedido {self.req[r].id} exige frio e foi recolhido pelo veículo {self.veh[v].id}")
            elif e["act"] == LOAD:
                if r in arrived:                              # carregado a partir do armazém
                    loaded[r] = e["t_start"]
                    load[v] += self.w[r]
                if self.req[r].cold and not self.veh[v].refrigerated:
                    problems.append(f"pedido {self.req[r].id} exige frio e foi carregado no veículo {self.veh[v].id}")
            elif e["act"] == DELIVER:
                if r in delivs: problems.append(f"pedido {self.req[r].id} entregue mais de uma vez")
                delivs[r] = e["t_start"]
                load[v] -= self.w[r]
            max_load[v] = max(max_load[v], load[v])
        off = self.off
        for r, q in enumerate(self.req):
            if r not in picks or r not in delivs:
                problems.append(f"pedido {q.id} não servido"); continue
            if not (q.a - off - EPS <= picks[r] <= q.a_end - off + EPS):
                problems.append(f"pedido {q.id}: recolha às {picks[r]:.0f} fora da janela")
            if delivs[r] < q.e - off - EPS:
                problems.append(f"pedido {q.id}: entregue antes de delivery_start")
            if delivs[r] < picks[r] + self.S + self.T[q.p][self.C] + self.H + self.T[self.C][q.d] - 1e-3:
                problems.append(f"pedido {q.id}: entrega fisicamente impossível")
            if r in loaded and loaded[r] - arrived[r] > self.dwell + 1e-3:
                problems.append(f"pedido {q.id}: excedeu o tempo máximo no armazém")
        for v in range(self.K):
            if max_load[v] > self.cap[v] + EPS:
                problems.append(f"capacidade do veículo {self.veh[v].id} excedida")
        dwell = [max(0.0, loaded[r] - arrived[r]) for r in loaded]
        return dict(ok=not problems, problemas=problems, km=res["km"], atraso_min=res["late_min"], custo=res["cost"],
                    pedidos_atrasados=res["late_by_request"], carga_max_kg=[round(x, 1) for x in max_load],
                    max_armazem_h=round(max(dwell) / 60, 2) if dwell else 0.0)

    def timeline(self, routes, origin=None):
        """Cronograma paragem a paragem como DataFrame (com data e hora se origin for dado)."""
        res = self.simulate(routes, trace=True)
        rows = [dict(veiculo=e["vehicle"], acao=e["act"], pedido=(self.req[e["r"]].id if e["r"] is not None else None),
                     local=e["loc"], inicio=e["t_start"], fim=e["t_end"], km=round(e["km"], 2),
                     atraso_min=round(e["late"], 1), carga_kg=round(e["load"], 1)) for e in res["log"]]
        out = pd.DataFrame(rows)
        out["pedido"] = out["pedido"].astype("Int64")
        if origin is not None:
            fmt = lambda t: (origin + pd.Timedelta(minutes=t)).strftime("%d/%m %H:%M")
            out["hora_inicio"] = [fmt(t) for t in out.inicio]
            out["hora_fim"] = [fmt(t) for t in out.fim]
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Leitura dos dados
# ----------------------------------------------------------------------------------------------------------------------
def load_problem(data_dir="data_tiny", late_penalty=LATE_PENALTY, verbose=False):
    """
    Lê os CSV de data_dir (e rules.csv, se existir) e devolve (Problem, origin).
    Os instantes ficam em minutos desde origin: a meia noite de start_date (com regras) ou 1 dia antes da primeira janela
    de recolha (sem regras). Os veículos começam no tempo 0.
    """
    rules = load_rules(data_dir)
    rd = lambda f, **k: pd.read_csv(Path(data_dir) / f, **k)
    loc, prod, vehs, orders, dm = rd("locations.csv"), rd("products.csv"), rd("vehicles.csv"), rd("orders.csv"), rd("distance_matrix.csv")
    items = rd("order_items.csv", parse_dates=["availability_start", "availability_end", "delivery_start", "delivery_end"])
    items = items.merge(prod, on="product_id", suffixes=("", "_prod")).merge(orders, on="order_id")
    if "weight_kg" not in items.columns:
        items["weight_kg"] = items.quantity * items.unit_weight_kg
    cold_col = next((c for c in COLD_COLUMNS if c in prod.columns), None)
    items["cold"] = items[cold_col].astype(bool) if cold_col else False

    p_loc = loc[loc.location_type == "producer"].set_index("ref_id").location_id
    d_loc = loc[loc.location_type == "delivery_location"].set_index("ref_id").location_id
    c_loc = loc[loc.location_type == "collection_center"].set_index("ref_id").location_id   # center_id para location_id
    center = int(c_loc.iloc[0])

    if rules.start_date:
        origin = pd.Timestamp(rules.start_date).normalize()
    else:
        origin = items.availability_start.min() - pd.Timedelta(days=1)
    mins = lambda col: (col - origin).dt.total_seconds() / 60
    items["a"], items["a_end"], items["e"], items["l"] = (mins(items[c]) for c in
        ("availability_start", "availability_end", "delivery_start", "delivery_end"))

    m = int(loc.location_id.max()) + 1
    D, T = [[0.0] * m for _ in range(m)], [[0.0] * m for _ in range(m)]
    for r in dm.itertuples():
        D[r.origin_location_id][r.destination_location_id] = float(r.distance_km)
        T[r.origin_location_id][r.destination_location_id] = float(r.duration_min)

    items = items.sort_values("order_item_id")
    reqs = [Request(int(r.order_item_id), int(r.order_id), r.name, int(p_loc[r.producer_id]), int(d_loc[r.delivery_location_id]),
                    float(r.weight_kg), float(r.a), float(r.a_end), float(r.e), float(r.l), bool(r.cold))
            for r in items.itertuples()]
    refr = vehs.refrigerated.astype(bool) if "refrigerated" in vehs.columns else pd.Series(False, index=vehs.index)
    vs = [Vehicle(int(v.vehicle_id), float(v.capacity_kg), int(c_loc[v.center_id]), bool(f))
          for v, f in zip(vehs.itertuples(), refr)]
    P = Problem(reqs, vs, D, T, center, late_penalty, rules)
    if verbose:
        print(f"Dataset {data_dir}: regras de {rules.source}")
        print(f"  S = {P.S:g} min por paragem, H = {P.H:g} min no armazém, máximo no armazém = "
              f"{rules.max_warehouse_dwell_h:g} h, janelas sobre o {'início' if P.off == 0 else 'fim'} do serviço")
        print(f"  origem do tempo: {origin}")
        print(f"  {P.n} pedidos ({items.order_id.nunique()} encomendas, {sum(P.w):.1f} kg), {P.K} veículos "
              + ", ".join(f"[{v.id}: {v.cap:g} kg{', refrigerado' if v.refrigerated else ''}]" for v in vs)
              + f", centro = location {P.C}, pedidos que exigem frio: {sum(q.cold for q in reqs)}")
    return P, origin


# ----------------------------------------------------------------------------------------------------------------------
# Teste rápido:  python model.py [pasta_dos_dados]
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "data_tiny"
    P, origin = load_problem(data, verbose=True)
    print(P.rules.table().to_string(index=False))
    late = {r.id: round(P.lateness_best_case(r), 1) for r in P.req if P.lateness_best_case(r) > 0}
    print("Pedidos que não cumprem delivery_end nem no melhor caso (min de atraso):", late)

    # Solução ingénua (só para testar o modelo): por ordem de início da recolha, cada pedido (P, L, D) vai para o fim da
    # rota do primeiro veículo em que a solução continua viável.
    pos = {q.id: i for i, q in enumerate(P.req)}
    ids, routes, unserved = [], [[] for _ in range(P.K)], []
    for r in sorted(range(P.n), key=lambda r: P.req[r].a):
        rid = P.req[r].id
        for v in P.allowed[r]:
            trial = [list(rt) for rt in routes]
            trial[v] += [(PICK, rid), (LOAD, rid), (DELIVER, rid)]
            Q = P.subset(ids + [rid])
            qi = {q.id: i for i, q in enumerate(Q.req)}
            if Q.simulate([[(a, None if i is None else qi[i]) for a, i in rt] for rt in trial]):
                routes, ids = trial, ids + [rid]
                break
        else:
            unserved.append(rid)
    Q = P.subset(ids)
    qi = {q.id: i for i, q in enumerate(Q.req)}
    routes = [[(a, None if i is None else qi[i]) for a, i in rt] for rt in routes]
    print(f"Rota ingénua: {len(ids)} de {P.n} pedidos servidos; por servir: {sorted(unserved)}")
    res = Q.simulate(routes)
    print("Custo:", {k: round(v, 1) for k, v in res.items() if k in ("cost", "km", "late_min")})
    print("Auditoria:", {k: v for k, v in Q.audit(routes).items() if k != "pedidos_atrasados"})
