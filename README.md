
# CMB.TECH Seasonality Project (Power BI)

Portfolio project testing whether shipping stocks show reliable seasonal
cycles, split into two independent pre-merger lineages: Euronav (tankers)
and Golden Ocean (dry bulk), ahead of their 2024/2025 mergers into
CMB.TECH.

## Status

Data acquisition complete. Power BI model in progress.

## Structure

The tree splits on ownership rather than on processing stage. Everything under
`data/` is this project's own output and is committed; everything under
`vendor/` is somebody else's data, kept locally so the committed series can be
rebuilt, and is gitignored in full. Each folder under `data/` carries a
`_manifest.json` recording where its series came from, how it was built and a
SHA-256 per file, which is what ties a committed series back to the capture
behind it.

- `data/equity/tanker/` — five crude tanker series, 2005 onward
- `data/equity/drybulk/` — seven dry bulk series, 2015 onward, plus the
  retired `GOGLO.ST` kept as a documented data-quality finding
- `data/freight/` — BDI (1985-), BDTI (2001-), BCI (2012-), daily
- `data/fleet/` — CMB.TECH vessel register snapshot
- `data/dayrates.csv` — disclosed day rates by vessel class, 2013-2026
- `scripts/` — acquisition and parsing, driven from the notebook
- `notebooks/` — the driver notebook
- `powerbi/` — the .pbix file
- `screenshots/` — report screenshots and GIFs for this README
- `vendor/` — gitignored; vendor captures and exports

## Hypothesis

See the full implementation plan for the two-lineage test design and the
decision branch (void vs. build a forward-looking blended model).