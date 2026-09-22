"""SGP4 self-check, frame invariance, hand-off statistics."""
import json
import pathlib

import numpy as np
import pytest
import sgp4
from sgp4.api import Satrec

from tlv import tle_ephem as te

SGP4_DIR = pathlib.Path(sgp4.__file__).parent
REAL_ROOT = pathlib.Path("data/real")


def test_sgp4_matches_bundled_verification_vectors():
    """Propagate the first verification TLE and compare to the package's own tcppver.out."""
    lines = [ln for ln in (SGP4_DIR / "SGP4-VER.TLE").read_text().splitlines()
             if not ln.startswith("#")]
    sat = Satrec.twoline2rv(lines[0][:69], lines[1][:69])

    expected = []
    for ln in (SGP4_DIR / "tcppver.out").read_text().splitlines()[1:4]:
        f = ln.split()
        expected.append([float(x) for x in f[:7]])

    for tsince, *state in expected:
        jd = sat.jdsatepoch
        fr = sat.jdsatepochF + tsince / 1440.0
        e, r, v = sat.sgp4(jd, fr)
        assert e == 0
        assert np.allclose(r, state[:3], atol=1e-6), (tsince, r, state[:3])
        assert np.allclose(v, state[3:], atol=1e-9), (tsince, v, state[3:])


def test_parse_tles_sorts_and_dedupes():
    text = (SGP4_DIR / "SGP4-VER.TLE").read_text()
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    doubled = "\n".join(lines[:2] + lines[:2] + lines[2:4])
    sats = te.parse_tles(doubled)
    epochs = [jd for jd, _ in sats]
    assert len(sats) == 2                      # duplicate collapsed
    assert epochs == sorted(epochs)


def test_frame_rotation_preserves_magnitudes_and_sma():
    """|r|, |v| and the semi-major axis are invariant under TEME -> J2000."""
    from qutils.orbital import ECI2OE

    lines = [ln for ln in (SGP4_DIR / "SGP4-VER.TLE").read_text().splitlines()
             if not ln.startswith("#")]
    sat = Satrec.twoline2rv(lines[0][:69], lines[1][:69])
    fr = sat.jdsatepochF + np.arange(10) * 60.0 / 86400.0
    e, r, v = sat.sgp4_array(np.full(10, float(sat.jdsatepoch)), fr)
    assert not np.any(e)

    r2, v2 = te.teme_to_j2000(r, v, sat.jdsatepoch + fr)
    assert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(r2, axis=1), rtol=1e-9)

    # Velocity is NOT magnitude-invariant to 1e-9: TEME rotates w.r.t. GCRS at the
    # precession rate (~7.7e-12 rad/s), so the transform adds omega x r ~ 5e-8 km/s,
    # i.e. ~6e-9 relative. The sma inherits twice that through v^2.
    assert np.allclose(np.linalg.norm(v, axis=1), np.linalg.norm(v2, axis=1), rtol=1e-7)

    a1 = np.array([ECI2OE(a, b)[0] for a, b in zip(r, v)])
    a2 = np.array([ECI2OE(a, b)[0] for a, b in zip(r2, v2)])
    assert np.allclose(a1, a2, rtol=1e-7)


def test_segment_ephemeris_uses_each_tle_inside_its_own_window():
    """Every sample takes the most recent TLE not later than that sample."""
    lines = [ln for ln in (SGP4_DIR / "SGP4-VER.TLE").read_text().splitlines()
             if not ln.startswith("#")]
    sats = te.parse_tles("\n".join(lines[:2]))
    # one TLE only: a 10-minute ephemeris, no hand-offs
    sats = [sats[0], (sats[0][0] + 10 / 1440.0, sats[0][1])]
    r, v, t_jd, idx, jump = te.segment_ephemeris(sats)
    assert r.shape == (11, 3) and jump.shape == (11,)
    assert np.all(np.diff(t_jd) > 0)
    assert idx[0] == 0 and idx[-1] == 1
    assert jump[idx.tolist().index(1)] == 0.0    # same TLE object -> exactly zero jump


def _real_dirs():
    return sorted(REAL_ROOT.glob("*/*/meta.json")) if REAL_ROOT.exists() else []


@pytest.mark.skipif(not _real_dirs(), reason="no real dataset built yet (scripts/build_real_dataset.py)")
@pytest.mark.parametrize("meta_path", _real_dirs(), ids=lambda p: str(p.parent))
def test_handoff_statistics_gate(meta_path):
    """Hand-off gate: median jump < 2 km, p95 < 10 km. Otherwise the window (then the
    object) is unsuitable."""
    meta = json.loads(meta_path.read_text())
    assert meta["handoff_median_km"] < 2.0, meta
    assert meta["handoff_p95_km"] < 10.0, meta
