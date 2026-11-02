import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from src.graph.categories import ALL_NODES, NODE_INSTITUTIONAL
from src.graph.diffusion import (
    adjacency_matrix,
    build_graph,
    diffuse,
    estimate_lag,
    laplacian,
)

_A, _NODES = adjacency_matrix(build_graph())
_L = laplacian(_A)
origins = st.sampled_from(ALL_NODES)
alphas = st.floats(0.01, 2.0)


@given(origins, st.floats(0.0, 100.0), alphas)
def test_diffusion_conserves_total_signal(origin: str, t: float, alpha: float) -> None:
    s0 = np.zeros(len(_NODES))
    s0[_NODES.index(origin)] = 1.0
    assert abs(diffuse(_L, s0, t, alpha).sum() - 1.0) < 1e-9


@given(origins, st.floats(0.0, 100.0), alphas)
def test_diffusion_never_produces_negative_concentration(
    origin: str, t: float, alpha: float
) -> None:
    s0 = np.zeros(len(_NODES))
    s0[_NODES.index(origin)] = 1.0
    assert diffuse(_L, s0, t, alpha).min() >= -1e-9


@given(origins, alphas)
def test_time_zero_is_the_identity(origin: str, alpha: float) -> None:
    s0 = np.zeros(len(_NODES))
    s0[_NODES.index(origin)] = 1.0
    assert np.allclose(diffuse(_L, s0, 0.0, alpha), s0)


@given(origins, alphas)
def test_estimated_lag_is_the_first_half_peak_crossing_of_the_institutional_node(
    origin: str, alpha: float
) -> None:
    """Recompute the documented contract independently of ``estimate_lag``."""
    n_steps, t_max = 60, 48.0
    s0 = np.zeros(len(_NODES))
    s0[_NODES.index(origin)] = 1.0
    times = np.linspace(0.0, t_max, n_steps)
    target = _NODES.index(NODE_INSTITUTIONAL)
    conc = np.array([diffuse(_L, s0, t, alpha)[target] for t in times])
    expected = t_max if conc.max() == 0 else float(times[np.argmax(conc >= 0.5 * conc.max())])
    assert estimate_lag(origin, alpha=alpha, n_steps=n_steps, t_max=t_max) == expected


def test_institutional_origin_is_already_at_half_peak_at_time_zero() -> None:
    assert estimate_lag(NODE_INSTITUTIONAL) == 0.0
