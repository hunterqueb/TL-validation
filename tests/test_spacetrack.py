"""The .env loader and the cache-first rule (no network in the test suite)."""
import os

import pytest

from tlv import spacetrack as st


def test_load_env_parses_comments_quotes_and_export(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# a comment\n\n'
                   'SPACETRACK_USER="quoted-user"\n'
                   "export SPACETRACK_PASS='exported-pass'\n"
                   "BLANK\n")
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASS", raising=False)

    st.load_env(env)
    assert os.environ["SPACETRACK_USER"] == "quoted-user"
    assert os.environ["SPACETRACK_PASS"] == "exported-pass"
    assert "BLANK" not in os.environ


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SPACETRACK_USER=from-file\nSPACETRACK_PASS=from-file\n")
    monkeypatch.setenv("SPACETRACK_USER", "from-environment")
    monkeypatch.setenv("SPACETRACK_PASS", "from-environment")

    st.load_env(env)
    assert st.credentials() == ("from-environment", "from-environment")


def test_credentials_says_what_to_do_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASS", raising=False)
    monkeypatch.setattr(st, "ENV_FILE", tmp_path / "does-not-exist")

    with pytest.raises(RuntimeError, match="SPACETRACK_USER and SPACETRACK_PASS not set"):
        st.credentials()


def test_cache_hit_needs_no_credentials_and_no_network(tmp_path, monkeypatch):
    """The cache is the unit of reproducibility: a second call must not hit the network."""
    cache = tmp_path / "tle"
    st.cache_path(99999, "2026-01-01", "2026-01-02", cache).parent.mkdir(parents=True)
    st.cache_path(99999, "2026-01-01", "2026-01-02", cache).write_text("1 99999U ...\n")

    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.setattr(st.requests, "Session", lambda: pytest.fail("hit the network"))

    text = st.fetch_tle_history(99999, "2026-01-01", "2026-01-02", cache)
    assert text.startswith("1 99999U")
    assert st.sha256(text) == st.sha256("1 99999U ...\n")
