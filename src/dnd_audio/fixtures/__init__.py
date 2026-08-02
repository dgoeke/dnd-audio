"""The synthetic fixture generator.

The spec is explicit that this project builds fixtures rather than checking audio
binaries into the repository, and M1 is where that generator lands. It is a shipped
package rather than test-local code because M2 through M5 are all tested against the
same fixtures, and `scripts/make_fixture.py` materializes one on disk for manual runs.

Three modules, deliberately separate:

* :mod:`dnd_audio.fixtures.wav` assembles RIFF/RF64 bytes. It knows nothing about
  sessions.
* :mod:`dnd_audio.fixtures.synth` produces deterministic signals. It knows nothing
  about files.
* :mod:`dnd_audio.fixtures.session` places signals on a timeline, writes the files and
  the `session.yaml`, and returns the ground truth a test asserts against.
"""

from __future__ import annotations

from dnd_audio.fixtures.session import (
    ClapInterval,
    FixtureChunk,
    FixtureSession,
    FixtureTrack,
    FixtureTruth,
    SpeechInterval,
    WrittenChunk,
    build_session,
    canonical_session,
)

__all__ = [
    "ClapInterval",
    "FixtureChunk",
    "FixtureSession",
    "FixtureTrack",
    "FixtureTruth",
    "SpeechInterval",
    "WrittenChunk",
    "build_session",
    "canonical_session",
]
