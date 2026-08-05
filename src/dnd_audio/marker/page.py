"""The standalone phone page: a transport for canonical bytes, never a second synthesizer.

The charter is unambiguous about what this file is and is not. Python creates the canonical
WAV once; **the same byte sequence** is written as `.wav` and encoded into the HTML; the
JavaScript builds a `Blob` from those bytes and plays it. No Web Audio oscillator, no
`Math.sin`, no lossy re-encode, no second base64 copy kept elsewhere in source. Extracting the
page's payload must produce the same length and the same SHA-256 as the CLI's WAV — which is
true here by construction rather than by testing, because both come from one call to
:func:`~dnd_audio.marker.wav.marker_wav_bytes`.

**Isolation is an allowlist and a policy, not a denylist.** The first draft of M10's plan
proposed enumerating network APIs to grep for; the second plan review pointed out that such a
list already missed CSS ``url(...)``, ``navigator.sendBeacon``, form actions and media/iframe
attributes — and would miss whatever browsers add next. So the page declares a
`Content-Security-Policy` of ``default-src 'none'`` plus exactly what it needs, and the test
parses the document and asserts every URL-bearing attribute is absent or a `blob:`/`data:` the
page generated itself.

**The playback state machine is data.** It sits in the page as a JSON transition table that
the page's own JavaScript reads to decide what any event does — so the default-suite test
parses *the same table the page runs on* and asserts no user event can start a second
playback while one is in progress. A test that grepped for a boolean guard would be asserting
something about the shape of the source rather than about the behaviour.

What that still cannot prove — that the JavaScript applies the table, that `ended` resets the
UI, that the download yields the canonical WAV — belongs to the physical bench, and the
charter's completion gate says so rather than implying a software proof that does not exist.
"""

from __future__ import annotations

import base64
import html
import json
from typing import Any, Final

from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.marker import MARKER_SAMPLE_RATE, artifact_stem
from dnd_audio.marker.spec import MarkerSpec
from dnd_audio.marker.wav import BITS_PER_SAMPLE, CHANNELS

__all__ = [
    "CONTENT_SECURITY_POLICY",
    "PAYLOAD_ELEMENT_ID",
    "STATE_MACHINE",
    "STATE_MACHINE_ELEMENT_ID",
    "marker_page_html",
    "payload_from_html",
]

#: The element the canonical WAV is embedded in, and the one a test extracts from.
PAYLOAD_ELEMENT_ID: Final = "marker-payload"

#: The element holding the transition table. Read by the page's script and by the test.
STATE_MACHINE_ELEMENT_ID: Final = "marker-state-machine"

#: ``default-src 'none'`` and then only what the page actually uses. ``'unsafe-inline'`` is
#: required for the page's own inline script and style and is not a loosening here: there is
#: no external origin the policy could otherwise admit, and no remote code to confuse with
#: local code. ``media-src blob:`` is what lets an object URL built from the embedded bytes
#: reach the audio element.
CONTENT_SECURITY_POLICY: Final = (
    "default-src 'none'; "
    "script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; "
    "media-src blob:; "
    "connect-src 'none'; "
    "img-src 'none'; "
    "font-src 'none'; "
    "frame-src 'none'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

#: The page's playback contract, as data.
#:
#: ``press`` is the only user-initiated event, and it is deliberately **absent** from the
#: ``on`` map of every busy state. That absence is the whole guarantee: the script consults
#: this table and does nothing for an event a state does not handle, so a second press during
#: playback cannot start anything. Asserted in the default suite against this exact document.
STATE_MACHINE: Final[dict[str, Any]] = {
    "initial": "idle",
    "userEvents": ["press"],
    #: Entering one of these begins playback. Nothing else may.
    "startsPlaybackOnEntry": ["playing"],
    #: Audio is sounding or about to. No user event may be handled here.
    "busyStates": ["counting", "playing"],
    "states": {
        "idle": {"on": {"press": "counting"}},
        "counting": {"on": {"countdownFinished": "playing", "playbackFailed": "idle"}},
        "playing": {"on": {"ended": "finished", "playbackFailed": "idle"}},
        "finished": {"on": {"press": "counting"}},
    },
}

_COUNTDOWN_SECONDS: Final = 3


def marker_page_html(spec: MarkerSpec, wav_bytes: bytes) -> str:
    """The complete standalone page for ``spec``, carrying ``wav_bytes`` exactly once.

    Args:
        spec: What is being played, for the identity the page displays.
        wav_bytes: The canonical WAV. Passed in rather than rebuilt so that the page and the
            `.wav` file are provably the same bytes rather than two calls that ought to agree.
    """
    payload = base64.b64encode(wav_bytes).decode("ascii")
    digest = sha256_bytes(wav_bytes)
    stem = artifact_stem(spec.name)
    seconds = spec.total_samples / MARKER_SAMPLE_RATE
    chirps = ", ".join(
        f"{chirp.start_hz}&nbsp;&rarr;&nbsp;{chirp.end_hz}&nbsp;Hz" for chirp in spec.chirps
    )
    gaps = ", ".join(f"{gap * 1000 // MARKER_SAMPLE_RATE}&nbsp;ms" for gap in spec.gaps_samples)

    return _TEMPLATE.format(
        policy=html.escape(CONTENT_SECURITY_POLICY, quote=True),
        stem=html.escape(stem),
        name=html.escape(spec.name),
        rationale=html.escape(spec.rationale),
        digest=html.escape(digest),
        seconds=f"{seconds:.3f}",
        chirps=chirps,
        gaps=gaps,
        rate=MARKER_SAMPLE_RATE,
        channels=CHANNELS,
        bits=BITS_PER_SAMPLE,
        countdown=_COUNTDOWN_SECONDS,
        payload_id=PAYLOAD_ELEMENT_ID,
        machine_id=STATE_MACHINE_ELEMENT_ID,
        machine=canonical_json(STATE_MACHINE).strip(),
        payload=payload,
    )


def payload_from_html(text: str) -> bytes:
    """The WAV bytes embedded in ``text``, decoded.

    The inverse the equivalence test runs, and deliberately written here rather than in the
    test: a test that carried its own extractor could pass against a page nothing else can
    read. Anything that consumes these artifacts uses this function.

    Raises:
        ValueError: if the payload element is absent or appears more than once. More than one
            is as much a failure as none — the charter requires the canonical payload to be
            present *exactly* once, because a second copy is a second thing that can drift.
    """
    opening = f'<script type="application/json" id="{PAYLOAD_ELEMENT_ID}">'
    occurrences = text.count(opening)
    if occurrences != 1:
        message = (
            f"expected exactly one {PAYLOAD_ELEMENT_ID} element, found {occurrences}. "
            f"A second copy of the payload is a second thing that can drift from the WAV."
        )
        raise ValueError(message)

    start = text.index(opening) + len(opening)
    end = text.index("</script>", start)
    return base64.b64decode(json.loads(text[start:end]))


#: Doubled braces are literal ones for `str.format`; every single-brace pair is substituted.
_TEMPLATE: Final = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{policy}">
<title>{stem}</title>
<style>
:root {{ color-scheme: dark; }}
body {{
  margin: 0; padding: 1.5rem; background: #111318; color: #e8eaed;
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-text-size-adjust: 100%;
}}
h1 {{ font-size: 1.15rem; margin: 0 0 .25rem; }}
p.sub {{ margin: 0 0 1.25rem; color: #9aa0a6; font-size: .9rem; }}
button {{
  width: 100%; min-height: 6.5rem; font-size: 1.6rem; font-weight: 600;
  color: #06210f; background: #7ee2a8; border: 0; border-radius: 14px;
  touch-action: manipulation; cursor: pointer;
}}
button:disabled {{ background: #33383f; color: #8b9097; cursor: default; }}
#status {{
  margin: 1.1rem 0; padding: .9rem 1rem; border-radius: 12px;
  background: #1b1f26; font-size: 1.15rem; font-weight: 600; text-align: center;
}}
dl {{ display: grid; grid-template-columns: auto 1fr; gap: .3rem .9rem;
     margin: 1.25rem 0 0; font-size: .82rem; }}
dt {{ color: #9aa0a6; }}
dd {{ margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
a {{ color: #7ee2a8; }}
ol {{ padding-left: 1.2rem; color: #9aa0a6; font-size: .85rem; }}
</style>
</head>
<body>
<h1>Sync marker &mdash; {name}</h1>
<p class="sub">{rationale}</p>

<button id="play" type="button">Play marker</button>
<div id="status" role="status" aria-live="polite">Ready</div>

<ol>
  <li>Media volume at the step recorded in the bench log. Bluetooth off.</li>
  <li>Phone flat, screen up, at the fixed central position. Do not move it between the
      start and end markers &mdash; a moved phone measures geometry, not clocks.</li>
  <li>Wait for silence in the room, then press once and leave it alone.</li>
</ol>

<dl>
  <dt>Marker</dt><dd>{name}</dd>
  <dt>SHA-256</dt><dd><code>{digest}</code></dd>
  <dt>Duration</dt><dd>{seconds} s</dd>
  <dt>Format</dt><dd>{rate} Hz, {channels} ch, {bits}-bit PCM</dd>
  <dt>Chirps</dt><dd>{chirps}</dd>
  <dt>Gaps</dt><dd>{gaps}</dd>
  <dt>Played</dt><dd><span id="count">0</span> time(s) this session</dd>
  <dt>Download</dt><dd><a id="download" download="{stem}.wav">{stem}.wav</a></dd>
</dl>

<script type="application/json" id="{machine_id}">{machine}</script>
<script type="application/json" id="{payload_id}">"{payload}"</script>
<script>
// The page reads its own transition table rather than hard-coding the rules, so the test
// that parses this element is asserting the behaviour and not a resemblance to it.
var MACHINE = JSON.parse(document.getElementById({machine_id!r}).textContent);
var PAYLOAD = JSON.parse(document.getElementById({payload_id!r}).textContent);

var button = document.getElementById("play");
var status = document.getElementById("status");
var counter = document.getElementById("count");

var raw = atob(PAYLOAD);
var bytes = new Uint8Array(raw.length);
for (var i = 0; i < raw.length; i++) {{ bytes[i] = raw.charCodeAt(i); }}
// One Blob, one object URL, used by both the player and the download link. A second
// encoding anywhere would be a second copy of the payload.
var url = URL.createObjectURL(new Blob([bytes], {{ type: "audio/wav" }}));
document.getElementById("download").href = url;

var audio = new Audio();
audio.src = url;
audio.preload = "auto";

var state = MACHINE.initial;
var played = 0;
var ticking = null;

function send(event) {{
  var handled = MACHINE.states[state].on;
  if (!Object.prototype.hasOwnProperty.call(handled, event)) {{ return false; }}
  state = handled[event];
  render();
  if (MACHINE.startsPlaybackOnEntry.indexOf(state) !== -1) {{ begin(); }}
  return true;
}}

function busy() {{ return MACHINE.busyStates.indexOf(state) !== -1; }}

function render() {{
  button.disabled = busy();
  if (state === "counting") {{ return; }}
  if (state === "playing") {{ status.textContent = "Playing\\u2026 keep still"; }}
  else if (state === "finished") {{ status.textContent = "Finished"; }}
  else {{ status.textContent = "Ready"; }}
}}

function begin() {{
  var attempt = audio.play();
  if (attempt && typeof attempt.catch === "function") {{
    attempt.catch(function () {{
      status.textContent = "Playback refused \\u2014 tap again";
      send("playbackFailed");
    }});
  }}
}}

function countdown(remaining) {{
  if (remaining <= 0) {{ ticking = null; send("countdownFinished"); return; }}
  status.textContent = String(remaining);
  ticking = setTimeout(function () {{ countdown(remaining - 1); }}, 1000);
}}

button.addEventListener("click", function () {{
  // Every playback begins from a user gesture; mobile autoplay is never assumed.
  if (send("press")) {{ countdown({countdown}); }}
}});

audio.addEventListener("ended", function () {{
  played += 1;
  counter.textContent = String(played);
  send("ended");
}});

audio.addEventListener("error", function () {{
  if (ticking !== null) {{ clearTimeout(ticking); ticking = null; }}
  status.textContent = "Could not play the embedded audio";
  send("playbackFailed");
}});

render();
</script>
</body>
</html>
"""
