# Contributing

Thanks for looking under the hood. This repo implements five textbook
quant models from scratch — correctness and honest reporting matter
more than features.

## Run the tests first
```bash
pip install -e .
python -m pytest -q
```
Every change must keep the full suite green, and new behaviour needs
new tests. Each model has a parameter-recovery test: simulate from
known parameters, and the estimator must find them back. If a change
breaks recovery, the estimator is wrong — not the test.

## Standards held in this repo
- **From scratch, textbook-faithful** — closed forms and estimators
  are implemented from the papers/textbooks, not wrapped from a
  quant library, and the docstrings cite the source.
- **Real data or labelled synthetic** — say which, with provenance
  and the fetch route recorded in the case file.
- **Numbers in `results/` must be reproducible** by the case scripts.
  If a change alters a reported number, regenerate the results file
  in the same change and say why.
- **Limitations are documented, not buried** — weak identification,
  regime breaks, measure mismatches (P vs Q) and data quirks are
  reported in the results, not footnoted away.
- **Seeded and typed** — every simulation takes a seed; public
  functions carry type hints and docstrings.

## Pull requests
Small, single-purpose PRs with a description of what changed and what
the tests show. Discussion first for anything that changes an
estimator, a calibration sample, or a headline number.
