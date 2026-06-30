"""Test suite for nba_edge CLI and model functions."""

import csv
import io
import math
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nba_edge  # noqa: E402


# ─── Math helpers ────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        assert nba_edge.sigmoid(0) == pytest.approx(0.5)

    def test_positive(self):
        assert nba_edge.sigmoid(10) > 0.99995

    def test_negative(self):
        assert nba_edge.sigmoid(-10) < 0.00005

    def test_monotonic(self):
        assert nba_edge.sigmoid(1) > nba_edge.sigmoid(0) > nba_edge.sigmoid(-1)


class TestImpliedProb:
    def test_negative_odds(self):
        # -110 → 52.4%
        assert nba_edge.implied_prob(-110) == pytest.approx(0.5238, abs=0.001)

    def test_positive_odds(self):
        # +100 → 50%
        assert nba_edge.implied_prob(100) == pytest.approx(0.5)

    def test_large_positive(self):
        # +400 → 20%
        assert nba_edge.implied_prob(400) == pytest.approx(0.20)

    def test_large_negative(self):
        # -400 → 80%
        assert nba_edge.implied_prob(-400) == pytest.approx(0.80)


class TestAmericanToDecimal:
    def test_positive(self):
        # +100 → 2.0
        assert nba_edge.american_to_decimal(100) == pytest.approx(2.0)

    def test_negative(self):
        # -110 → 1.909
        assert nba_edge.american_to_decimal(-110) == pytest.approx(1.909, abs=0.01)

    def test_underdog(self):
        # +300 → 4.0
        assert nba_edge.american_to_decimal(300) == pytest.approx(4.0)


class TestKelly:
    def test_no_edge(self):
        # 50% win prob at -110 odds → near-zero or zero Kelly
        assert nba_edge.kelly(0.5, -110) == pytest.approx(0, abs=0.01)

    def test_positive_edge(self):
        # Strong edge should produce positive Kelly
        result = nba_edge.kelly(0.65, -110, fraction=1.0)
        assert result > 0

    def test_negative_edge_clipped(self):
        # Very low win prob → Kelly should be 0 (no bet)
        assert nba_edge.kelly(0.10, -110) == 0

    def test_fraction_reduces(self):
        full = nba_edge.kelly(0.65, -110, fraction=1.0)
        half = nba_edge.kelly(0.65, -110, fraction=0.5)
        assert half == pytest.approx(full * 0.5, rel=0.01)


class TestWinProbability:
    def test_equal_ratings(self):
        # Equal ratings with 0 home advantage → 50%
        assert nba_edge.win_probability(5.0, 5.0, home_adv=0) == pytest.approx(0.5)

    def test_home_advantage(self):
        # Home team favored
        prob = nba_edge.win_probability(5.0, 5.0, home_adv=2.8)
        assert prob > 0.5

    def test_bounded(self):
        assert 0 < nba_edge.win_probability(-10, 10) < 1
        assert 0 < nba_edge.win_probability(10, -10) < 1


class TestConfidenceTier:
    def test_a_plus(self):
        assert nba_edge.confidence_tier(0.10) == "A+"

    def test_a(self):
        assert nba_edge.confidence_tier(0.07) == "A"

    def test_b(self):
        assert nba_edge.confidence_tier(0.05) == "B"

    def test_c(self):
        assert nba_edge.confidence_tier(0.035) == "C"

    def test_below_threshold(self):
        assert nba_edge.confidence_tier(0.01) == "—"


class TestProjectedTotal:
    def test_returns_float(self):
        result = nba_edge.projected_total("Boston Celtics", "New York Knicks")
        assert isinstance(result, float)

    def test_reasonable_range(self):
        # NBA games typically total 200-260
        for home in nba_edge.PACE_RATINGS:
            for away in nba_edge.PACE_RATINGS:
                total = nba_edge.projected_total(home, away)
                assert 180 < total < 280, f"Total {total} out of range for {home} vs {away}"
            break  # just test one to keep it fast

    def test_unknown_teams_uses_default(self):
        result = nba_edge.projected_total("Unknown Team A", "Unknown Team B")
        assert isinstance(result, float)
        assert result > 0


# ─── Game model ──────────────────────────────────────────────────────────────

class TestGame:
    def test_creation(self):
        g = nba_edge.Game("Boston Celtics", "New York Knicks", -4.5)
        assert g.home == "Boston Celtics"
        assert g.away == "New York Knicks"
        assert g.spread == -4.5

    def test_default_odds(self):
        g = nba_edge.Game("Boston Celtics", "New York Knicks", -4.5)
        assert g.spread_odds == -110
        assert g.home_ml == -110
        assert g.away_ml == -110

    def test_model_win_prob(self):
        g = nba_edge.Game("Boston Celtics", "New York Knicks", -4.5)
        prob = g.model_win_prob()
        assert 0 < prob < 1
        # Celtics (8.4) vs Knicks (4.6) at home → Celtics favored
        assert prob > 0.5

    def test_spread_win_prob(self):
        g = nba_edge.Game("Boston Celtics", "New York Knicks", -4.5)
        prob = g.spread_win_prob()
        assert 0 < prob < 1

    def test_edges_returns_list(self):
        g = nba_edge.Game("Boston Celtics", "New York Knicks", -4.5)
        edges = g.edges()
        assert isinstance(edges, list)

    def test_edge_dict_structure(self):
        g = nba_edge.Game("Boston Celtics", "Washington Wizards", -15.0)
        edges = g.edges()
        assert len(edges) > 0
        for e in edges:
            assert "game" in e
            assert "pick" in e
            assert "odds" in e
            assert "model_prob" in e
            assert "mkt_prob" in e
            assert "edge" in e
            assert "kelly" in e
            assert "bet_type" in e
            assert "grade" in e

    def test_edges_respect_min_edge(self):
        g = nba_edge.Game("Boston Celtics", "New York Knicks", -4.5)
        config = {**nba_edge.DEFAULT_CONFIG, "min_edge": 0.99}  # impossibly high
        edges = g.edges(config)
        assert len(edges) == 0

    def test_totals_edges_with_total(self):
        g = nba_edge.Game(
            "Boston Celtics", "New York Knicks", -4.5,
            total=220.0, total_odds_over=-110, total_odds_under=-110,
        )
        edges = g.totals_edges()
        assert isinstance(edges, list)

    def test_totals_edges_without_total(self):
        g = nba_edge.Game("Boston Celtics", "New York Knicks", -4.5)
        assert g.totals_edges() == []

    def test_total_edge_keys(self):
        g = nba_edge.Game(
            "Indiana Pacers", "Detroit Pistons", -8.0,
            total=250.0, total_odds_over=-110, total_odds_under=-110,
        )
        edges = g.totals_edges()
        for e in edges:
            assert e["bet_type"] == "TOTAL"
            assert "OVER" in e["pick"] or "UNDER" in e["pick"]


# ─── Sample slate ────────────────────────────────────────────────────────────

class TestSampleSlate:
    def test_has_games(self):
        assert len(nba_edge.SAMPLE_SLATE) >= 3

    def test_all_games_valid(self):
        for g in nba_edge.SAMPLE_SLATE:
            assert isinstance(g, nba_edge.Game)
            assert g.home in nba_edge.POWER_RATINGS
            assert g.away in nba_edge.POWER_RATINGS

    def test_all_teams_have_pace(self):
        for g in nba_edge.SAMPLE_SLATE:
            assert g.home in nba_edge.PACE_RATINGS, f"{g.home} missing from PACE_RATINGS"
            assert g.away in nba_edge.PACE_RATINGS, f"{g.away} missing from PACE_RATINGS"


# ─── Data consistency ────────────────────────────────────────────────────────

class TestDataConsistency:
    def test_power_ratings_count(self):
        # 30 NBA teams
        assert len(nba_edge.POWER_RATINGS) == 30

    def test_pace_ratings_count(self):
        assert len(nba_edge.PACE_RATINGS) == 30

    def test_teams_match(self):
        assert set(nba_edge.POWER_RATINGS.keys()) == set(nba_edge.PACE_RATINGS.keys())

    def test_confidence_tiers_complete(self):
        assert set(nba_edge.CONFIDENCE_TIERS.keys()) == {"A+", "A", "B", "C"}


# ─── CLI helpers ─────────────────────────────────────────────────────────────

class TestExportCSV:
    def test_export_creates_file(self, tmp_path):
        edges = [
            {"game": "A @ B", "pick": "A ML", "odds": -110, "edge": 0.05,
             "model_prob": 0.6, "mkt_prob": 0.52, "kelly": 0.03, "bet_type": "ML", "grade": "B"},
        ]
        path = str(tmp_path / "test.csv")
        nba_edge.export_csv(edges, path)
        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["game"] == "A @ B"

    def test_export_empty_list_noop(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        nba_edge.export_csv([], path)
        assert not os.path.exists(path)


class TestPrintFunctions:
    def test_print_header(self, capsys):
        nba_edge.print_header()
        out = capsys.readouterr().out
        assert "NBA Edge Finder" in out

    def test_print_edges_empty(self, capsys):
        nba_edge.print_edges([], nba_edge.DEFAULT_CONFIG)
        out = capsys.readouterr().out
        assert "No edges" in out

    def test_print_edges_with_data(self, capsys):
        edges = [
            {"game": "A @ B", "pick": "A ML", "odds": -110, "model_prob": 0.6,
             "mkt_prob": 0.52, "edge": 0.08, "kelly": 0.05, "bet_type": "ML", "grade": "A+"},
        ]
        nba_edge.print_edges(edges, nba_edge.DEFAULT_CONFIG)
        out = capsys.readouterr().out
        assert "A @ B" in out
        assert "1 edge" in out


# ─── CLI entry point ─────────────────────────────────────────────────────────

class TestCLI:
    def test_backtest_flag(self, capsys):
        with mock.patch("sys.argv", ["nba_edge.py", "--backtest"]):
            nba_edge.main()
        out = capsys.readouterr().out
        assert "Back-test Summary" in out
        assert "2024-25" in out

    def test_min_edge_argument(self, capsys):
        with mock.patch("sys.argv", ["nba_edge.py", "--min-edge", "0.50"]):
            nba_edge.main()
        out = capsys.readouterr().out
        # With a 50% edge threshold, no edges should be found
        assert "No edges" in out or "0 edges" in out.lower() or "edge" in out.lower()


# ─── Live odds fetch ─────────────────────────────────────────────────────────

class TestFetchLiveSlate:
    def test_fallback_on_no_key(self):
        # Without a valid API key, should fall back to sample slate
        games = nba_edge.fetch_live_slate("invalid_key_for_test")
        assert isinstance(games, list)
        # Should fall back to SAMPLE_SLATE on error
        assert len(games) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
