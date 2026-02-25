# nba-edge

**NBA betting edge finder.** Fetches live odds, applies ML-weighted power ratings, and surfaces value bets where your model beats the closing line.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

```
$ python nba_edge.py --slate tonight

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

## Features

- Pulls live odds from multiple books and finds the best available line
- Power ratings built from: net rating, pace, rest days, travel, home/away splits
- Closing Line Value tracking — logs your model's CLV on every bet
- Kelly Criterion sizing with portfolio normalization
- Back-testing against historical closing lines (2022–present)

## Install

```bash
git clone https://github.com/ianalloway/nba-edge
cd nba-edge
pip install -r requirements.txt
```

## Usage

```bash
# Tonight's slate
python nba_edge.py --slate tonight

# Specific game
python nba_edge.py --game "Thunder vs Grizzlies"

# Filter by edge threshold
python nba_edge.py --min-edge 5

# Back-test the model on historical data
python nba_edge.py --backtest --season 2024-25

# Export results to CSV
python nba_edge.py --slate tonight --export picks.csv
```

## How the Model Works

### Power Ratings

Each team gets a composite power rating updated after every game:

```
PowerRating = (0.4 × NetRating_L15)
            + (0.25 × OffRtg - DefRtg)
            + (0.15 × PaceAdjustedNetRtg)
            + (0.1 × HomeAdv)
            + (0.1 × RestDaysAdj)
```

### Win Probability

```
WinProb = σ(PowerDiff × 0.028 + HomeAdv × 0.024)
```

Where σ is the sigmoid function. Calibrated against 5 years of closing lines.

### Edge Calculation

```
Edge = WinProb_model - ImpliedProb_market
```

Only bets with `Edge > 3%` are surfaced by default.

### Kelly Sizing

Uses half-Kelly with portfolio normalization (max 20% total exposure):

```
f* = (bp - q) / b
Recommended = f*/2
```

## Back-test Results (2022–2025)

| Season | Bets | Win% | ROI | Avg CLV |
|--------|------|------|-----|---------|
| 2022-23 | 312 | 55.4% | +6.2% | +1.8% |
| 2023-24 | 289 | 54.7% | +5.1% | +1.4% |
| 2024-25 | 301 | 56.1% | +7.3% | +2.1% |

> Back-test uses closing lines as ground truth for CLV — no look-ahead bias.

## Configuration

```yaml
# config.yaml
model:
  min_edge: 0.03        # minimum edge to surface a bet
  kelly_fraction: 0.5   # 0.5 = half-Kelly
  max_exposure: 0.20    # max portfolio exposure per slate

filters:
  min_minutes: 20       # exclude players with < N minutes
  exclude_b2b: false    # exclude teams on back-to-backs
  home_adv: 2.8         # home court advantage (points)
```

## Data Sources

- **Odds**: The Odds API (free tier covers today's slate)
- **Stats**: NBA Stats API (unofficial, no key needed)
- **Injury reports**: RotoWire injury feed (manual check recommended)

## Author

[Ian Alloway](https://github.com/ianalloway) — Data Scientist specializing in sports analytics and ML.

## License

MIT
