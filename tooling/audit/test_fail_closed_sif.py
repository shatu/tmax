"""Fail-closed local-SIF resolution (tmax-private#1).

Regression tests for the defect that ended trainer 11096823 at step 78.

What happened: `prefer_local_sif()` returned the bare image reference when a
configured SIF pool did not contain the image. The sandfleet worker then did a
per-lease `docker://` pull plus OCI->SIF conversion, which takes 600-1200 s and
fails `exit=255` under restricted egress. 58 of the 1,170 task images the run
touched were absent from the pool, so this presented as a fleet-wide restart
latency cliff (3.0 s -> 639 s median within five minutes) rather than as a
missing file, and cost a 6.5-hour 64-GPU run.

The tests assert the property that was missing, not the implementation:
a configured pool is authoritative, and a miss must be loud.

Red on canonical da146695 (which silently returns the bare ref); green with the
guard. Run:
    PYTHONPATH=<tree> python3 -m pytest test_fail_closed_sif.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from open_instruct.environments import apptainer_images as ai

IMAGE = "hamishi740/swerl-tmax-v3:deadbeefcafe"


def _pool(tmp: str, *names: str) -> str:
    for n in names:
        p = Path(tmp) / n
        p.write_bytes(b"\0" * 4096)   # non-empty, so _is_nonempty_file passes
    return tmp


class TestFailClosedSif(unittest.TestCase):
    def test_a_hit_still_resolves_to_the_local_sif(self):
        """The guard must not change the happy path."""
        with TemporaryDirectory() as tmp:
            name = ai._sif_name_for_image(IMAGE)
            _pool(tmp, name)
            with patch.dict("os.environ", {"SWERL_APPTAINER_SIF_DIR": tmp}, clear=False):
                got = ai.prefer_local_sif(IMAGE)
            self.assertTrue(got.endswith(".sif"), f"expected a local SIF, got {got!r}")
            self.assertIn(tmp, got)

    def test_b_miss_raises_instead_of_returning_a_remote_ref(self):
        """The core property: a miss must fail loudly, never fall back.

        NOTE on structure: an earlier version put `self.fail(...)` INSIDE
        `assertRaises(Exception)`. `self.fail` raises AssertionError, which is
        an Exception, so assertRaises swallowed the very signal the test
        existed to emit and the suite passed green against the unguarded tree.
        Call first, classify after — never assert-raises around your own fail.
        """
        import os
        with TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items()
                   if k != "SWERL_ALLOW_REMOTE_IMAGE_FALLBACK"}
            env["SWERL_APPTAINER_SIF_DIR"] = tmp
            outcome = err = None
            with patch.dict("os.environ", env, clear=True):
                try:
                    outcome = ai.prefer_local_sif(IMAGE)
                except Exception as e:          # noqa: BLE001 - classify below
                    err = e
            if err is None:
                self.fail(
                    f"prefer_local_sif returned {outcome!r} on a pool miss "
                    "instead of raising — this is the silent docker:// "
                    "fallback that killed trainer 11096823"
                )
            self.assertIn("deadbeefcafe", str(err), "error must name the missing image")

    def test_c_no_pool_configured_is_unchanged(self):
        """Without a configured pool there is nothing to be authoritative about."""
        import os
        env = {k: v for k, v in os.environ.items() if k != "SWERL_APPTAINER_SIF_DIR"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(ai.prefer_local_sif(IMAGE), IMAGE)

    def test_d_explicit_opt_out_restores_fallback(self):
        """An operator can still choose remote pulls, but must say so."""
        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ",
                            {"SWERL_APPTAINER_SIF_DIR": tmp,
                             "SWERL_ALLOW_REMOTE_IMAGE_FALLBACK": "1"}, clear=False):
                self.assertEqual(ai.prefer_local_sif(IMAGE), IMAGE)

    def test_e_empty_sif_counts_as_a_miss(self):
        """A zero-byte SIF is not coverage.

        The audit gate nearly shipped with 'non-empty + inspect' as the validity
        test; a truncated file would have satisfied it. Resolution must agree.
        """
        with TemporaryDirectory() as tmp:
            name = ai._sif_name_for_image(IMAGE)
            (Path(tmp) / name).write_bytes(b"")      # zero length
            import os
            env = {k: v for k, v in os.environ.items()
                   if k != "SWERL_ALLOW_REMOTE_IMAGE_FALLBACK"}
            env["SWERL_APPTAINER_SIF_DIR"] = tmp
            outcome = err = None
            with patch.dict("os.environ", env, clear=True):
                try:
                    outcome = ai.prefer_local_sif(IMAGE)
                except Exception as e:          # noqa: BLE001 - classify below
                    err = e
            if err is None:
                self.fail(f"empty SIF accepted as a hit / fell back: {outcome!r}")

    def test_f_absolute_sif_paths_bypass_resolution(self):
        """Callers that already hold a path must be untouched by the guard."""
        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"SWERL_APPTAINER_SIF_DIR": tmp}, clear=False):
                self.assertEqual(ai.prefer_local_sif("/abs/x.sif"), "/abs/x.sif")
                self.assertEqual(ai.prefer_local_sif("./rel.sif"), "./rel.sif")


if __name__ == "__main__":
    unittest.main(verbosity=2)
