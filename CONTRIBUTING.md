# Contributing to nba-edge

This repo is a legacy single-file CLI demo — the maintained library lives at
[nba-ratings](https://github.com/ianalloway/nba-ratings). If you're looking to
contribute real model/library improvements, go there instead. Small fixes to
this demo CLI are still welcome here.

## Getting Started

1. **Fork** this repo and create your branch from `main`
2. Make your changes with clear, descriptive commits
3. **Test** your changes locally before opening a PR (`python -m pytest`)
4. Open a Pull Request describing what and why

## Development Setup

Pure Python standard library — no dependencies, no `requirements.txt` needed.

```bash
git clone https://github.com/ianalloway/nba-edge
cd nba-edge
python nba_edge.py --slate sample
python -m pytest
```

## Code Style

- Python 3.11+, stdlib only. Don't add a runtime dependency without discussing it first.
- Keep functions small and focused — one job per function.
- Write self-documenting code; add comments only where logic is non-obvious.

## Pull Request Guidelines

- Keep PRs focused — one fix per PR
- Include a clear description of **what** and **why**
- Reference related issues with `Closes #123`
- All CI checks must pass before merging

## Reporting Bugs

Open an issue using the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) with steps to reproduce, expected vs actual behavior, and your Python version.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Questions? Open an issue or reach out: **ian@allowayllc.com**
