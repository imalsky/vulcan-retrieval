"""The v5 model options together on the real smoke pipeline: sulfur free, the Kzz
power law, T at 1 bar as the deep parameter with the adiabat below a free
boundary, the Gaussian mass nuisance and the grey deck. One invariant: the
pipeline builds, its layout is the engine's, a prior draw evaluates through the
cold batched evaluator with three-wide references, and the v4 layout is
unchanged when every option is off. Slow (full tier): builds and compiles the
real chemistry twice."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402

pytestmark = pytest.mark.slow


def test_v5_options_build_sample_and_evaluate(v5_smoke_pipe, smoke_pipe):
    pipe = v5_smoke_pipe
    cfg = pipe.cfg
    # the layout: the engine's chemistry block, then the T-P block with T1bar and
    # the boundary, lnR0, the Gaussian lnM, the deck
    assert pipe.n_chem == 5 and pipe.fwd.chem.chem_extras == ("s_o", "kzz_slope")
    assert pipe.names[:pipe.n_chem_tp] == ["lnZ", "c_o", "lnKzz", "s_o", "kzz_slope",
                                           "Tirr", "log10kappa", "T1bar", "log10P_rcb"]
    assert pipe.names[pipe.n_chem_tp:pipe.n_chem_tp + 3] == ["lnR0", "lnM", "log10P_deck"]
    assert pipe.fwd.tp_model.deep_param == "T1bar" and pipe.fwd.tp_model.adiabat
    assert pipe.fwd.chem.kzz_mode == "powerlaw"
    assert next(s for s in C.specs_from_config(cfg, groups=pipe.groups)
                if s.name == "lnM").prior_type == "gaussian"
    # the window holds on the chemistry grid too: every accepted prior draw's
    # profile is inside it down to the chemistry bottom
    U = pipe.sample_prior_u(jax.random.PRNGKey(0), 4)
    Theta = np.asarray(jax.vmap(pipe.theta_from_u)(U))
    p_chem = np.asarray(pipe.fwd.p_chem_bar)
    for th in Theta:
        T = np.asarray(pipe.fwd.tp_model.eval(th[pipe.n_chem:pipe.n_chem_tp], p_chem))
        assert np.all(np.isfinite(T)) and T.min() >= pipe.fwd.tp_model.T_min
        assert T.max() <= pipe.fwd.tp_model.T_max
    # the cold batched evaluator: finite or the reject sentinel, references
    # (lnZ, c_o, s_o) of the kept rows, zeros elsewhere
    Y0, refs0 = P._blank_state(pipe, 2)
    assert refs0.shape == (2, 3)
    ev = jax.jit(pipe._make_batch_eval("cold"))
    L, Ynew, refs, st = ev(U[:2], Y0, refs0, np.ones(2, bool))
    L, refs = np.asarray(L), np.asarray(refs)
    assert L.shape == (2,) and refs.shape == (2, 3)
    kept = L > P.REJECT_BELOW
    assert np.all(np.isfinite(L[kept]))
    i_s = pipe.names.index("s_o")
    # the jitted transform and the eager one differ by an ulp
    np.testing.assert_allclose(refs[kept], Theta[:2][kept][:, [0, 1, i_s]], rtol=1e-12, atol=0)
    assert np.all(refs[~kept] == 0.0)
    # with every option off the v4 layout is unchanged
    base = smoke_pipe
    assert base.n_chem == 3 and base.fwd.chem.chem_extras == ()
    assert base.names[:3] == ["lnZ", "c_o", "lnKzz"] and "lnM" not in base.names
    assert base.cfg.cloud_model == "powerlaw" and base.n_cloud == 2
