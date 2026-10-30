# todo

The evidence behind each item is in `notes.md` (sections named below); solver and engine
items live in VULCAN3's and vulcan-forward's own `todo.md`.

1. [owed] Rewrite the three WASP-107b certificates on NAS with the report's fixed Kzz block (absolute Kzz, floor weights from the prior, no x10 ceiling off W39b's profile); their notes "Runs" numbers were derived by hand.

## 1. WASP-107b certificates

Evidence: notes "Design records" (the run certificate's `derived` block) and "Runs" (the
w107b entries, whose log10 Kzz and "above 10^11.6" weights were computed by hand). On
NAS, after `git pull --ff-only` and the bootstrap: add `"network":
"thermo/SNCHO_photo_network.txt"` to each run's `config.json` (the report refuses one
without it), then from the repo root `python -m retrieval_framework.nautilus_report
output/w107b/<run>_nautilus --spectra 0` for `w107b_joint`, `w107b_welbanks` and
`w107b_g395h`. Done when the three rewritten certificates' Kzz blocks are in "Runs".
