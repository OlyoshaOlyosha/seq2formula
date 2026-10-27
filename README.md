# seq2formula

**English** · [Русский](README.ru.md)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-%3E%3D0.110-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-99%20passed-3FC07A)
![Local](https://img.shields.io/badge/network-none-8B93F8)

seq2formula is a local experiment in symbolic regression: given a finite
sequence of numbers, the system searches for a closed-form expression f(n)
that generates it — `1 2 4 8 16` → `2^n`. The search currently runs as a
genetic programming loop, with a reinforcement-learning engine planned on
the same base; every generation is streamed live to a browser dashboard.

![seq2formula dashboard](docs/img/dashboard-search.png)

The project is a long-running personal experiment rather than a finished
product: the grammar, the engines and the dashboard are expected to evolve
iteratively, and results — including trained models, once the neuro engine
lands — are meant to be kept, compared and improved over time. Everything
runs locally: a Python engine does the searching, a browser page does the
showing, no network access is used or required.

## How it works

- Candidate formulas are expression trees over a small grammar: `n`,
  numeric constants (plus π and e), `+ - * / ^ %` (remainder), unary
  minus, and functions — `sqrt`, cube root, factorial, `abs`, `sin`,
  `cos`, `tan`, `ln`.
- You choose which operators and functions the search may use — turn any
  of them off and watch how (or whether) the evolution finds another
  route. Basic operators are on by default; the advanced set is one click
  away.
- Fitness combines **accuracy** (a candidate is evaluated on n = 1..N; the
  closer the values to the target sequence, the better) with **parsimony**
  (shorter formulas win ties, which keeps solutions readable and prevents
  oversized "technically correct" monsters).
- Answers are validated by values, not by text: `n+n` and `2*n` count as
  the same answer, and several different correct formulas for one sequence
  is a normal outcome.

## Dashboard

The search is observable end to end: current best formula (rendered with
KaTeX), a top-5 leaderboard of the current generation, a live event feed
(new records, exact hits, formula shrinking), fitness and error dynamics
per generation, population diversity, tree-size distribution, per-term
error tables, a history of solved puzzles, a hall of fame of the shortest
formulas, and a gallery of curious near-misses. A run can be paused,
stepped through generation by generation, or accelerated at any time.
An **Autopilot** mode can run through the entire puzzle library (65
sequences) automatically, solving them one by one in the background.

## Getting started

Requires Python 3.10+.

```bat
git clone https://github.com/OlyoshaOlyosha/seq2formula.git
cd seq2formula
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
run.bat
```

`run.bat` starts the local server and opens the dashboard in the browser.

## Status and directions

Experimental, in continuous development.

- Stage 1 (done) — genetic programming engine with a post-solution
  "shrink" phase, live dashboard, operator picker, built-in puzzle
  library of graded difficulty.
- Grammar v2 (done) — unary functions (roots, factorial, trigonometry,
  logarithm), the remainder operator, and the π / e constants, with the
  picker regrouped into basic and advanced sets; puzzle tiers T4/T5
  showcase the new operations (65 puzzles in total).
- Performance and search quality (done) — compiled-formula evaluation
  (~3× faster generations), diversity mechanics (fitness sharing,
  adaptive anti-stagnation shocks, weighted parsimony), and a settings
  sweep over the whole library; the live feed renders formulas with math
  typography, and any formula can be plotted next to its sequence.
- Stage 2 (next) — reinforcement-learning engine that assembles formulas
  step by step and is trained live, per puzzle; trained policies are
  saved, reused and compared.
- Later — recurrences, more statistics, more tuning experiments.

## Notes

- The dashboard UI is Russian — the project is developed for personal use
  first.
- No license yet; the decision is deliberately deferred.
