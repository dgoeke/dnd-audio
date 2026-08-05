"""The standalone page: the same bytes, no network, and one playback at a time.

Three claims, and it matters which is proved how.

**Byte equivalence** is mechanical and complete: the payload is extracted, decoded and
compared to the CLI's WAV. It is also true by construction — both come from one call to
`marker_wav_bytes` — so this is a guard against someone later introducing a second encoding,
which is exactly the failure the charter's "not two approximately equivalent synthesizers"
rule exists to prevent.

**Isolation** is proved by parsing rather than by grepping. Every URL-bearing attribute in the
document must be absent or a `blob:`/`data:` the page generated itself, and the page must
carry a restrictive CSP. The second plan review was right that a denylist of known network
APIs already missed CSS ``url(...)``, ``sendBeacon``, form actions and media attributes — and
would miss whatever comes next.

**Playback state** is proved against the declarative transition table *the page's own script
reads*, so the assertion is about behaviour rather than about a resemblance in the source.
What that still cannot prove — that the JavaScript applies the table, that `ended` resets the
UI, that the download yields the canonical WAV — belongs to the physical bench, and the
charter's amended completion gate says so.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest

from dnd_audio.determinism import sha256_bytes
from dnd_audio.marker.page import (
    CONTENT_SECURITY_POLICY,
    PAYLOAD_ELEMENT_ID,
    STATE_MACHINE,
    STATE_MACHINE_ELEMENT_ID,
    marker_page_html,
    payload_from_html,
)
from dnd_audio.marker.spec import MARKER_SPECS, MarkerSpec
from dnd_audio.marker.wav import marker_wav_bytes

ALL_SPECS = pytest.mark.parametrize("spec", MARKER_SPECS.values(), ids=list(MARKER_SPECS))

#: Attributes that can cause a browser to fetch something. Checked exhaustively against the
#: parsed document rather than searched for in the text: the point is to enumerate what the
#: page *has*, not to guess what it might have.
URL_ATTRIBUTES = frozenset(
    {
        "action",
        "background",
        "cite",
        "codebase",
        "data",
        "formaction",
        "href",
        "icon",
        "manifest",
        "ping",
        "poster",
        "profile",
        "src",
        "srcset",
        "usemap",
    }
)

#: Schemes a self-contained page may legitimately reference. `blob:` is the object URL the
#: script builds from the embedded bytes; `data:` would be an inline payload. Anything else
#: leaves the device.
LOCAL_SCHEMES = ("blob:", "data:", "#")


class _Attributes(HTMLParser):
    """Every ``(tag, attribute, value)`` in the document, including inside inline blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[tuple[str, str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            self.found.append((tag, name.lower(), value))

    handle_startendtag = handle_starttag


def parse(text: str) -> list[tuple[str, str, str | None]]:
    parser = _Attributes()
    parser.feed(text)
    return parser.found


@pytest.fixture(scope="module", params=list(MARKER_SPECS), ids=list(MARKER_SPECS))
def page(request: pytest.FixtureRequest) -> tuple[MarkerSpec, bytes, str]:
    spec = MARKER_SPECS[request.param]
    wav = marker_wav_bytes(spec)
    return spec, wav, marker_page_html(spec, wav)


class TestTheEmbeddedBytesAreTheCanonicalBytes:
    """The milestone's central equivalence claim."""

    def test_the_extracted_payload_is_byte_identical_to_the_wav(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        _, wav, text = page
        assert payload_from_html(text) == wav

    def test_and_has_the_same_digest(self, page: tuple[MarkerSpec, bytes, str]) -> None:
        _, wav, text = page
        assert sha256_bytes(payload_from_html(text)) == sha256_bytes(wav)

    def test_the_payload_appears_exactly_once(self, page: tuple[MarkerSpec, bytes, str]) -> None:
        """A second copy is a second thing that can drift from the WAV."""
        _, _, text = page
        assert text.count(f'id="{PAYLOAD_ELEMENT_ID}"') == 1

    def test_extraction_refuses_a_page_carrying_two_payloads(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        """The extractor is the guard, so it must fail rather than take the first."""
        _, _, text = page
        opening = f'<script type="application/json" id="{PAYLOAD_ELEMENT_ID}">'
        doubled = text.replace(opening, opening + '"AAAA"</script>' + opening, 1)
        with pytest.raises(ValueError, match="exactly one"):
            payload_from_html(doubled)

    def test_extraction_refuses_a_page_carrying_none(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            payload_from_html("<html><body>nothing here</body></html>")

    def test_the_page_contains_no_second_synthesis(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        """No oscillator, no trigonometry, no second waveform. The charter's core rule."""
        _, _, text = page
        for forbidden in (
            "OscillatorNode",
            "createOscillator",
            "AudioContext",
            "webkitAudioContext",
            "Math.sin",
            "Math.cos",
            "createBuffer",
        ):
            assert forbidden not in text

    def test_the_identity_is_visible_to_the_operator(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        """Someone holding the phone must be able to say which marker this is."""
        spec, wav, text = page
        assert spec.name in text
        assert sha256_bytes(wav) in text


class TestNothingLeavesTheDevice:
    """Isolation as an allowlist plus a policy, never a list of known APIs."""

    def test_every_url_bearing_attribute_is_local(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        """Parsed and enumerated, so an attribute nobody thought of is still covered."""
        _, _, text = page
        offenders = [
            (tag, name, value)
            for tag, name, value in parse(text)
            if name in URL_ATTRIBUTES and value and not value.startswith(LOCAL_SCHEMES)
        ]
        assert offenders == []

    def test_the_document_carries_no_url_attribute_at_all(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        """Stronger than the previous test, and it is what the page actually achieves.

        Not one URL-bearing attribute is present in the served markup — even the download
        link's `href` is absent, because the script assigns it an object URL built from the
        embedded bytes at runtime. So a page loaded with scripting disabled fetches nothing
        because there is nothing to fetch, rather than because every target happens to be
        local. "Every URL attribute is local" would stay true if someone added a second one;
        this notices.
        """
        _, _, text = page
        carriers = {(tag, name) for tag, name, _ in parse(text) if name in URL_ATTRIBUTES}
        assert carriers == set()

    def test_no_external_scheme_appears_anywhere_in_the_document(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        """Belt and braces over the raw text, including inside CSS and script."""
        _, _, text = page
        assert not re.search(r"https?://", text)
        assert not re.search(r"""["'(]//[a-z0-9]""", text, re.IGNORECASE)

    def test_no_css_url_reference(self, page: tuple[MarkerSpec, bytes, str]) -> None:
        """The case the first draft's denylist missed."""
        _, _, text = page
        assert "url(" not in text
        assert "@import" not in text

    def test_the_content_security_policy_denies_by_default(
        self, page: tuple[MarkerSpec, bytes, str]
    ) -> None:
        _, _, text = page
        policies = [
            value
            for tag, name, value in parse(text)
            if tag == "meta" and name == "content" and value and "default-src" in value
        ]
        assert len(policies) == 1
        assert policies[0] == CONTENT_SECURITY_POLICY
        assert "default-src 'none'" in policies[0]
        assert "connect-src 'none'" in policies[0]
        assert "form-action 'none'" in policies[0]

    def test_there_is_no_form_and_no_frame(self, page: tuple[MarkerSpec, bytes, str]) -> None:
        tags = {tag for tag, _, _ in parse(text=page[2])}
        assert tags.isdisjoint({"form", "iframe", "frame", "object", "embed", "link", "img"})

    def test_no_service_worker_or_manifest(self, page: tuple[MarkerSpec, bytes, str]) -> None:
        """Explicit non-goals: no PWA, no service worker, no installability."""
        _, _, text = page
        for forbidden in (
            "serviceWorker",
            "navigator.sendBeacon",
            "manifest.json",
            "importScripts",
        ):
            assert forbidden not in text


class TestOnePlaybackAtATime:
    """Asserted against the transition table the page's own script reads."""

    def test_the_table_is_embedded_and_parses(self, page: tuple[MarkerSpec, bytes, str]) -> None:
        _, _, text = page
        opening = f'<script type="application/json" id="{STATE_MACHINE_ELEMENT_ID}">'
        start = text.index(opening) + len(opening)
        embedded = json.loads(text[start : text.index("</script>", start)])
        assert embedded == STATE_MACHINE

    def test_no_user_event_can_start_playback_while_busy(self) -> None:
        """The guarantee, stated over the graph rather than over the source.

        `press` is the only user-initiated event, and it must be unhandled in every busy
        state. Unhandled is stronger than "handled but ignored": the script does nothing for
        an event a state does not list, so absence *is* the guard.
        """
        for state in STATE_MACHINE["busyStates"]:
            transitions = STATE_MACHINE["states"][state]["on"]
            for event in STATE_MACHINE["userEvents"]:
                assert event not in transitions

    def test_playback_cannot_re_enter_itself_on_any_event(self) -> None:
        """No loop: an `ended` handler that returned to `playing` would repeat forever.

        Scoped to the states that *are* playing, not to every busy state — `counting` leads
        to `playing` on purpose, and asserting otherwise would forbid the page from ever
        making a sound.
        """
        starts = set(STATE_MACHINE["startsPlaybackOnEntry"])
        for state in starts:
            for target in STATE_MACHINE["states"][state]["on"].values():
                assert target not in starts

    def test_only_the_countdown_leads_into_playback(self) -> None:
        """Whatever else changes, playback is entered from exactly one place."""
        starts = set(STATE_MACHINE["startsPlaybackOnEntry"])
        entries = {
            (state, event)
            for state, definition in STATE_MACHINE["states"].items()
            for event, target in definition["on"].items()
            if target in starts
        }
        assert entries == {("counting", "countdownFinished")}

    def test_every_transition_target_is_a_declared_state(self) -> None:
        """A typo in a target would make the page wedge on that event."""
        states = set(STATE_MACHINE["states"])
        for definition in STATE_MACHINE["states"].values():
            assert set(definition["on"].values()) <= states

    def test_the_initial_state_is_idle_and_does_not_play(self) -> None:
        """Mobile autoplay is never assumed; playback begins from a gesture."""
        assert STATE_MACHINE["initial"] == "idle"
        assert STATE_MACHINE["initial"] not in STATE_MACHINE["startsPlaybackOnEntry"]

    def test_playback_is_reachable_from_idle_by_a_user_event(self) -> None:
        """The contrast that keeps the guarantees above from being vacuous.

        Every test in this class passes for a page whose button does nothing at all. This
        one says the machine can actually reach playback, and only through `press`.
        """
        after_press = STATE_MACHINE["states"]["idle"]["on"]["press"]
        assert after_press == "counting"
        assert (
            STATE_MACHINE["states"]["counting"]["on"]["countdownFinished"]
            in (STATE_MACHINE["startsPlaybackOnEntry"])
        )

    def test_a_failed_playback_returns_to_a_state_that_can_retry(self) -> None:
        """A phone refusing autoplay must not leave the page permanently stuck."""
        for state in STATE_MACHINE["busyStates"]:
            recovered = STATE_MACHINE["states"][state]["on"]["playbackFailed"]
            assert "press" in STATE_MACHINE["states"][recovered]["on"]


class TestDeterminism:
    """Two builds of one spec are the same page (INV-02)."""

    @ALL_SPECS
    def test_the_page_is_byte_stable(self, spec: MarkerSpec) -> None:
        wav = marker_wav_bytes(spec)
        assert marker_page_html(spec, wav) == marker_page_html(spec, wav)

    @ALL_SPECS
    def test_the_page_carries_no_clock_host_or_path(self, spec: MarkerSpec) -> None:
        """It is a deterministic artifact; anything per-run would break that."""
        text = marker_page_html(spec, marker_wav_bytes(spec))
        for forbidden in ("Date(", "new Date", "/home/", "/tmp/", "localhost"):
            assert forbidden not in text
