# IA26_P01_G13: agente de rotas para comida local

Projeto 01 de Inteligência Artificial (ESI, UPCA, 2026-27), grupo G13.

**Grupo:** Diogo Alexandre Alves da Silva (31504) · Gonçalo Rodrigues (31507) · Rui Barbosa (31515)

**Repositório:** https://github.com/disilva021P/IA26_P01_G13

## Conteúdo

| Ficheiro | Descrição |
|---|---|
| `01_Project01_G13.ipynb` | Notebook principal: introdução, resumo do EDA, modelação, pesquisa informada, pesquisa local, CSP, análise de resultados e conclusão |
| `02_eda_tiny.ipynb`, `02_eda_small.ipynb` | Análise exploratória de cada dataset (o mesmo código, só muda `DATA`) |
| `model.py` | Regras (`Rules`, `load_rules`), entidades (`Request`, `Vehicle`, `State`, `Problem`), ações, função de custo, simulador, verificador (`audit`) e leitura dos dados (`load_problem`) |
| `informed.py` | Greedy e A* (`best_first`), Greedy cronológico (`greedy_chrono`), heurísticas `h_chain` (h1), `h_disjoint` (h2) e `h_work` (h3) |
| `local_search.py` | Hill Climbing (`hill_climbing`) e vizinhança (`neighbours`, `apply_move`) |
| `csp.py` | CSP: variáveis, domínios, restrições, backtracking com MRV, forward checking, reinícios e branch & bound (`csp_solve`, `csp_complete`) |
| `agent.py` | Agente completo: grupos temporais, solução inicial e `solve_all` |
| `data_tiny/`, `data_small/` | Datasets (CSV), incluindo o `rules.csv` |
| `readme.txt` | Link do repositório |
| `requirements.txt` | Dependências Python |

## Como executar

```bash
pip install -r requirements.txt
python model.py data_tiny       # teste rápido do modelo (mostra as regras lidas)
jupyter notebook                # abrir os notebooks e usar "Run All"
```

Os notebooks leem os CSV de `data_tiny/` e `data_small/` (caminhos relativos), por isso devem ser executados a partir
desta pasta, com os módulos `.py` ao lado. Tempos de execução aproximados: `02_eda_*` cerca de 10 s cada;
`01_Project01_G13.ipynb` cerca de 9 min (o A* e o Greedy best first correm até aos limites de tempo nas instâncias grandes,
e o Hill Climbing no small demora cerca de 100 s).

Sem `rules.csv` numa pasta de dados, o modelo usa os valores por omissão (10 min de serviço, sem limite de tempo no
armazém, janelas sobre o início do serviço).
