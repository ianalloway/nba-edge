# nba-edge

> **This repo is a legacy CLI demo.** The maintained library (Elo ratings,
> logistic win probability, Kelly helpers — with tests and CI) lives at
> **[ianalloway/nba-ratings](https://github.com/ianalloway/nba-ratings)**
> (same PyPI name, `nba-edge`). Evaluation UI:
> **[nba-clv-dashboard](https://github.com/ianalloway/nba-clv-dashboard)**.
> Case study: [Sports ML evaluation](https://ianalloway.xyz/papers/sports-ml-evaluation-case-study.html).
>
> Go there for anything you'd actually depend on. What's left here is a
> single-file CLI toy kept around for the demo output below.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

## What this is

A ~450-line, dependency-free `nba_edge.py` script: hardcoded sample power
ratings, a sigmoid win-probability model, and a Kelly-sizing CLI that prints a
sample slate of "edges." No live data by default — `--slate live` will call
The Odds API if you pass a key, otherwise everything runs on the bundled
sample slate.

```
$ python nba_edge.py --slate sample

🏀 NBA Edge Finder — Feb 25 2026
════════════════════════════════════════════════════════

Game                    Pick            Odds    Model%  Mkt%    Edge    Kelly
──────────────────────────────────────────────────────────────────────────────
Thunder vs Grizzlies    Thunder -8      -110    64.2%   52.4%   +11.8%  5.6%
Lakers vs Warriors      Warriors ML     +145    46.8%   40.8%   +6.0%   3.2%
Celtics vs Knicks       Celtics -4.5    -108    58.1%   53.1%   +5.0%   2.7%

3 edges found above 3% threshold. Avg edge: +7.6%
Recommended exposure (half-Kelly portfolio): 5.8% of bankroll
```

## Install

No dependencies beyond the Python standard library — nothing to install.

```bash
git clone https://github.com/ianalloway/nba-edge
cd nba-edge
python nba_edge.py --slate sample
```

## Usage

```bash
python nba_edge.py --slate sample                 # bundled sample slate
python nba_edge.py --slate live --api-key KEY      # The Odds API (or set ODDS_API_KEY)
python nba_edge.py --min-edge 5                    # filter by edge threshold
python nba_edge.py --slate sample --export picks.csv
python nba_edge.py --backtest                      # prints illustrative demo numbers, NOT a real back-test
```

`--backtest` prints a fixed, made-up table to show the report shape — this
repo doesn't ship historical closing-line data, so there's no real back-test
to run. For an actual evaluation harness, see
[nba-clv-dashboard](https://github.com/ianalloway/nba-clv-dashboard).

## How the demo model works

Fixed sample power ratings per team, combined with a sigmoid win-probability
curve, compared against market-implied probability from the odds:

```
WinProb = σ(PowerDiff × 0.028 + HomeAdv × 0.024)
Edge    = WinProb_model − ImpliedProb_market
```

Only edges above `--min-edge` (default 3%) are shown, sized with half-Kelly
and a 20% max portfolio exposure cap.

## Tests

```bash
python -m pytest
```

## Author

[Ian Alloway](https://github.com/ianalloway) — Data Scientist specializing in sports analytics and ML.

## License

MIT
