# src/smoke_test.py
import asyncio
from pathlib import Path
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

TEAMS_DIR = Path(__file__).resolve().parent.parent / "teams"
TEAM_A_PATH = TEAMS_DIR / "team_a.txt"
TEAM_B_PATH = TEAMS_DIR / "team_b.txt"

def load_team(path: Path) -> str:
    txt = path.read_text(encoding="utf-8").strip()
    return txt + ("\n" if not txt.endswith("\n") else "")

async def main():
    team_a = load_team(TEAM_A_PATH)
    team_b = load_team(TEAM_B_PATH)

    p1 = RandomPlayer(
        battle_format="gen1ou",
        server_configuration=LocalhostServerConfiguration,
        team=team_a,
        log_level=30,  # quiet
    )
    p2 = RandomPlayer(
        battle_format="gen1ou",
        server_configuration=LocalhostServerConfiguration,
        team=team_b,
        log_level=30,
    )

    await p1.battle_against(p2, n_battles=3)
    print(f"Done. P1 wins: {p1.n_won_battles} / 3")

if __name__ == "__main__":
    asyncio.run(main())
