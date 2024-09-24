from __future__ import annotations

from pathlib import Path


def get_homogeneous_agent_paths(
    root_dir: Path,
    teacher: str,
    function: str,
) -> list[str]:
    root_path = root_dir / "ToySGD" / teacher
    # Sort directories to ensure same sequence for reproducibility
    agent_dirs = sorted(
        [
            entry.name
            for entry in root_path.iterdir()
            if entry.is_dir() and entry.name != "combined"
        ],
    )
    paths = []
    for dirname in agent_dirs:
        agent_path = root_path / dirname / function
        paths.append(agent_path)
    return paths
