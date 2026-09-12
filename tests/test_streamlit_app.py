"""
Smoke tests for app/streamlit_app.py using Streamlit's AppTest framework.
AppTest can't simulate a real file_uploader interaction (no API for that as
of this Streamlit version), so these tests cover what IS testable: the app
runs without raising, and shows the correct state before any upload / when
no checkpoint exists yet. Full upload -> analyze -> report flow is covered
by tests/test_pipeline.py (the underlying pipeline) and manual/API testing.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_runs_without_crashing_with_no_checkpoint():
    # Real config has no trained checkpoint in this fresh repo state, so the
    # app should show its "no checkpoint" error path, not raise an exception.
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception
    # Should show the "no trained model checkpoint" error message
    error_texts = " ".join(e.value for e in at.error)
    assert "No trained model checkpoint" in error_texts or len(at.error) == 0
    # (len==0 branch covers the case a checkpoint DOES exist in this env,
    # e.g. if a real Stage 3+ run left one behind — either way, no crash)


def test_app_title_renders():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception
    titles = [t.value for t in at.title]
    assert any("ImageForensics AI" in t for t in titles)
