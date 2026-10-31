# IA26 P01 (Trabalho de IA, UPCA 2026-27)

**Grupo:** Diogo Alexandre Alves da Silva (31504) · Gonçalo Rodrigues (31507) · Rui Barbosa (31515)

## Repositório
**Link:** [GitHub](https://github.com/disilva021P/IA26_P01_G13)

## Conteúdo
| Ficheiro | Descrição |
|---|---|
| `Proj01.ipynb` | Notebook principal: introdução, problema, modelação, resultados e conclusão |
| `EDA.ipynb` | Análise exploratória dos dados |
| `model.py` | Modelo do problema: pedidos, veículos, estado, ações, função de custo, simulador e verificador (`audit`) |
| `algoritmos.py` | Algoritmos: Greedy e A* (`best_first`), Hill Climbing e solução inicial (`initial_solution`, `temporal_groups`) |
| `data_tiny/` | Dataset (CSV, versão *tiny*): produtores, clientes, encomendas, veículos, matriz de distâncias, etc. |
| `requirements.txt` | Dependências Python |

## Como executar
```bash
pip install -r requirements.txt
jupyter notebook          # abrir EDA.ipynb e Proj01.ipynb e usar "Run All"
```
Os notebooks leem os CSV da pasta `data_tiny/` (caminho relativo), por isso devem ser executados a partir da pasta onde estão, com `model.py` e `algoritmos.py` ao lado.

