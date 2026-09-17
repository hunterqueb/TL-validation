"""Download TLE history from space-track.org and cache it on disk."""
import hashlib
import os
import pathlib
import time

import requests

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
QUERY_URL = ("https://www.space-track.org/basicspacedata/query/class/gp_history/"
             "NORAD_CAT_ID/{norad}/EPOCH/{start}--{stop}/orderby/EPOCH asc/format/tle")


def cache_path(norad, start, stop, cache_dir="data/tle"):
    return pathlib.Path(cache_dir) / f"{norad}_{start}_{stop}.tle"


def sha256(text):
    """sha256 of the cache file contents -- the unit of reproducibility (HANDOFF 3.1)."""
    return hashlib.sha256(text.encode()).hexdigest()


def fetch_tle_history(norad, start, stop, cache_dir="data/tle"):
    """Return the TLE text for one object in [start, stop].

    Read the cache first. Query space-track only when the cache file is absent.
    Dates are ISO strings, for example "2026-01-01". One request per (object, window):
    the rate limit is 30/min and 300/hour, so never loop over days.
    """
    path = cache_path(norad, start, stop, cache_dir)
    if path.exists():
        return path.read_text()

    session = requests.Session()
    resp = session.post(LOGIN_URL, data={"identity": os.environ["SPACETRACK_USER"],
                                         "password": os.environ["SPACETRACK_PASS"]})
    resp.raise_for_status()
    resp = session.get(QUERY_URL.format(norad=norad, start=start, stop=stop))
    if resp.status_code == 429:  # back off once, then give up -- do not hammer
        time.sleep(60)
        resp = session.get(QUERY_URL.format(norad=norad, start=start, stop=stop))
    resp.raise_for_status()
    text = resp.text
    if not text.strip():
        raise RuntimeError(f"space-track returned no TLEs for {norad} in {start}--{stop}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
