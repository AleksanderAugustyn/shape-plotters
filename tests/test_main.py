"""CLI arg parsing selects the right render without opening a window."""
from main import RENDERS
from src.renders.beta import BetaRender
from src.renders.fos import FoSRender


def test_render_registry() -> None:
    assert RENDERS == {"beta": BetaRender, "fos": FoSRender}
