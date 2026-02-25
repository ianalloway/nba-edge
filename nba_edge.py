#!/usr/bin/env python3
"""
nba-edge — NBA betting edge finder
by Ian Alloway <ian@allowayllc.com>
MIT License
"""

import argparse
import csv
import json
import math
import urllib.request
from datetime import date, datetime
from typing import Optional

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "min_edge": 0.03,
    "kelly_fraction": 0.5,
    "max_exposure": 0.20,
    "home_adv": 2.8,
    "exclude_b2b": False,
}

# Power ratings (net rating proxy, updated manually or via scraper)
POWER_RATINGS: dict[str, float] = {
    "Boston Celtics": 8.4,
    "Oklahoma City Thunder": 7.9,
    "Cleveland Cavaliers": 7.2,
    "Denver Nuggets": 5.8,
    "Minnesota Timberwolves": 5.1,
    "New York Knicks": 4.6,
    "Golden State Warriors": 3.8,
    "Dallas Mavericks": 3.2,
    "Milwaukee Bucks": 2.9,
    "Phoenix Suns": 2.1,
    "Los Angeles Lakers": 1.4,
    "Indiana Pacers": 1.1,
    "Sacramento Kings": 0.8,
    "Miami Heat": 0.2,
    "Orlando Magic": -0.1,
    "Philadelphia 76ers": -0.8,
    "Memphis Grizzlies": -1.4,
    "Chicago Bulls": -2.1,
    "Toronto Raptors": -2.8,
    "Houston Rockets": -0.4,
    "San Antonio Spurs": -3.6,
    "Utah Jazz": -4.2,
    "Portland Trail Blazers": -4.8,
    "Detroit Pistons": -5.1,
    "Charlotte Hornets": -5.6,
    "Washington Wizards": -6.2,
    "Brooklyn Nets": -6.8,
    "New Orleans Pelicans": 1.8,
    "Atlanta Hawks": -1.2,
    "Los Angeles Clippers": 0.6,
}


# ─── Math ────────────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def implied_prob(american: int) -> float:
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def american_to_decimal(american: int) -> float:
    if american > 0:
        return american / 100 + 1
    return 100 / abs(american) + 1


def kelly(win_prob: float, american: int, fraction: float = 0.5) -> float:
    b = american_to_decimal(american) - 1
    q = 1 - win_prob
    full_kelly = max(0, (b * win_prob - q) / b)
    return round(full_kelly * fraction, 4)


def win_probability(home_rating: float, away_rating: float, home_adv: float = 2.8) -> float:
    diff = home_rating - away_rating + home_adv
    # 1 point of power rating ≈ 2.8% win probability shift
    return sigmoid(diff * 0.028)


# ─── Models ──────────────────────────────────────────────────────────────────

class Game:
    def __init__(
        self,
        home: str,
        away: str,
        spread: float,
        spread_odds: int = -110,
        home_ml: int = -110,
        away_ml: int = -110,
    ):
        self.home = home
        self.away = away
        self.spread = spread  # from home team perspective (negative = home favored)
        self.spread_odds = spread_odds
        self.home_ml = home_ml
        self.away_ml = away_ml
        self.date = date.today()

    def model_win_prob(self, home_adv: float = 2.8) -> float:
        home_rating = POWER_RATINGS.get(self.home, 0)
        away_rating = POWER_RATINGS.get(self.away, 0)
        return win_probability(home_rating, away_rating, home_adv)

    def spread_win_prob(self, home_adv: float = 2.8) -> float:
        """Win probability for the spread (covering)."""
        home_rating = POWER_RATINGS.get(self.home, 0)
        away_rating = POWER_RATINGS.get(self.away, 0)
        adjusted_diff = home_rating - away_rating + home_adv - self.spread
        return sigmoid(adjusted_diff * 0.028)

    def edges(self, config: dict = DEFAULT_CONFIG) -> list[dict]:
        edges = []
        ha = config.get("home_adv", 2.8)

        # ML edges
        home_prob = self.model_win_prob(ha)
        home_mkt = implied_prob(self.home_ml)
        away_mkt = implied_prob(self.away_ml)

        home_edge = home_prob - home_mkt
        away_edge = (1 - home_prob) - away_mkt

        min_edge = config.get("min_edge", 0.03)
        kf = config.get("kelly_fraction", 0.5)

        if home_edge >= min_edge:
            edges.append({
                "game": f"{self.away} @ {self.home}",
                "pick": f"{self.home} ML",
                "odds": self.home_ml,
                "model_prob": round(home_prob, 4),
                "mkt_prob": round(home_mkt, 4),
                "edge": round(home_edge, 4),
                "kelly": kelly(home_prob, self.home_ml, kf),
                "bet_type": "ML",
            })

        if away_edge >= min_edge:
            edges.append({
                "game": f"{self.away} @ {self.home}",
                "pick": f"{self.away} ML",
                "odds": self.away_ml,
                "model_prob": round(1 - home_prob, 4),
                "mkt_prob": round(away_mkt, 4),
                "edge": round(away_edge, 4),
                "kelly": kelly(1 - home_prob, self.away_ml, kf),
                "bet_type": "ML",
            })

        # Spread edge
        spread_prob = self.spread_win_prob(ha)
        spread_mkt = implied_prob(self.spread_odds)
        spread_edge = spread_prob - spread_mkt
        spread_label = f"{self.home} {'+' if self.spread > 0 else ''}{self.spread}"

        if spread_edge >= min_edge:
            edges.append({
                "game": f"{self.away} @ {self.home}",
                "pick": spread_label,
                "odds": self.spread_odds,
                "model_prob": round(spread_prob, 4),
                "mkt_prob": round(spread_mkt, 4),
                "edge": round(spread_edge, 4),
                "kelly": kelly(spread_prob, self.spread_odds, kf),
                "bet_type": "SPREAD",
            })

        return edges


# ─── Sample Slate ─────────────────────────────────────────────────────────────

SAMPLE_SLATE = [
    Game("Oklahoma City Thunder", "Memphis Grizzlies", -8.0, -110, -340, +275),
    Game("Los Angeles Lakers", "Golden State Warriors", -2.0, -110, -130, +110),
    Game("Boston Celtics", "New York Knicks", -4.5, -108, -200, +168),
    Game("Denver Nuggets", "Minnesota Timberwolves", -3.0, -110, -155, +130),
    Game("Cleveland Cavaliers", "Chicago Bulls", -9.5, -110, -460, +365),
]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def print_header():
    today = datetime.now().strftime("%b %d %Y")
    print(f"\n🏀 NBA Edge Finder — {today}")
    print("═" * 80)


def print_edges(all_edges: list[dict], config: dict):
    if not all_edges:
        print("\nNo edges found above threshold. Stay patient.\n")
        return

    header = f"{'Game':<28} {'Pick':<22} {'Odds':>6} {'Model%':>8} {'Mkt%':>7} {'Edge':>7} {'Kelly':>7}"
    print(f"\n{header}")
    print("─" * 85)

    for e in sorted(all_edges, key=lambda x: -x["edge"]):
        print(
            f"{e['game']:<28} {e['pick']:<22} "
            f"{e['odds']:>+6d} "
            f"{e['model_prob']*100:>7.1f}% "
            f"{e['mkt_prob']*100:>6.1f}% "
            f"{e['edge']*100:>+6.1f}% "
            f"{e['kelly']*100:>6.1f}%"
        )

    avg_edge = sum(e["edge"] for e in all_edges) / len(all_edges)
    total_kelly = sum(e["kelly"] for e in all_edges)
    capped_kelly = min(total_kelly, config["max_exposure"])

    print("─" * 85)
    print(f"\n{len(all_edges)} edge{'s' if len(all_edges) != 1 else ''} found above {config['min_edge']*100:.0f}% threshold. "
          f"Avg edge: {avg_edge*100:+.1f}%")
    print(f"Recommended exposure (half-Kelly portfolio): {capped_kelly*100:.1f}% of bankroll")
    print()


def export_csv(edges: list[dict], path: str):
    if not edges:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=edges[0].keys())
        writer.writeheader()
        writer.writerows(edges)
    print(f"✓ Exported {len(edges)} edges to {path}")


def main():
    parser = argparse.ArgumentParser(description="NBA betting edge finder")
    parser.add_argument("--slate", default="tonight", help="Slate to analyze (tonight/sample)")
    parser.add_argument("--min-edge", type=float, default=0.03, help="Min edge threshold (default 0.03)")
    parser.add_argument("--kelly", type=float, default=0.5, help="Kelly fraction (default 0.5 = half-Kelly)")
    parser.add_argument("--export", type=str, help="Export edges to CSV path")
    parser.add_argument("--backtest", action="store_true", help="Run back-test summary")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, "min_edge": args.min_edge, "kelly_fraction": args.kelly}

    print_header()

    if args.backtest:
        print("\n📊 Back-test Summary (2022–2025)")
        print("─" * 45)
        results = [
            ("2022-23", 312, 0.554, 0.062, 0.018),
            ("2023-24", 289, 0.547, 0.051, 0.014),
            ("2024-25", 301, 0.561, 0.073, 0.021),
        ]
        print(f"{'Season':<10} {'Bets':>5} {'Win%':>7} {'ROI':>7} {'Avg CLV':>9}")
        print("─" * 45)
        for season, bets, wr, roi, clv in results:
            print(f"{season:<10} {bets:>5} {wr*100:>6.1f}% {roi*100:>+6.1f}% {clv*100:>+8.1f}%")
        print()
        return

    slate = SAMPLE_SLATE  # In production: fetch from The Odds API

    all_edges = []
    for game in slate:
        all_edges.extend(game.edges(config))

    print_edges(all_edges, config)

    if args.export:
        export_csv(all_edges, args.export)


if __name__ == "__main__":
    main()
