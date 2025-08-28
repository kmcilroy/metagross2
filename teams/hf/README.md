Metamon Teams (Hugging Face)

This folder holds Pokémon Showdown team files fetched from the Hugging Face dataset `jakegrigsby/metamon-teams`.

Quick start (PowerShell):

1) Install deps once:
   - pip install huggingface_hub
2) List available bundles:
   - python scripts/hf_download_teams.py --list
3) Download a subset (example: Gen 1 OU competitive):
   - python scripts/hf_download_teams.py --groups competitive --tiers gen1ou
4) Or fetch everything (large):
   - python scripts/hf_download_teams.py --all

Files are extracted under teams/hf/<group>/<tier>/.

Notes:
- Public dataset; no token required. For private mirrors set HF_TOKEN env var.
- To refresh, re-run with --overwrite.
