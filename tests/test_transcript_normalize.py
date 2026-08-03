"""The only text processing in this project, and the spec's hard limit on it.

*"Do not add an LLM prose-cleanup pass in the MVP. Preserve what ASR produced, with only
deterministic whitespace/punctuation normalization necessary for rendering."* These tests are
mostly about what is **not** done: the sentence that comes out is the sentence that went in,
with its spacing fixed and nothing else.
"""

from __future__ import annotations

import pytest

from dnd_audio.transcript.normalize import (
    comparison_key,
    normalize_text,
    similarity_permille,
    word_count,
)


class TestNormalizationIsForRendering:
    def test_runs_of_whitespace_collapse(self) -> None:
        assert normalize_text("We   should\tgo") == "We should go"

    def test_newlines_collapse_so_markdown_stays_one_line_per_turn(self) -> None:
        assert normalize_text("We should\ngo back") == "We should go back"

    def test_surrounding_whitespace_goes(self) -> None:
        assert normalize_text("  hello  ") == "hello"

    def test_a_space_before_closing_punctuation_goes(self) -> None:
        assert normalize_text("Really ? Yes .") == "Really? Yes."

    def test_unicode_is_put_in_one_form(self) -> None:
        """Two spellings of the same character would otherwise make one transcript differ
        from another byte for byte (INV-02)."""
        composed = "Zephyriné"  # e + combining acute
        assert normalize_text(composed) == normalize_text("Zephyriné")

    def test_it_is_idempotent(self) -> None:
        """What makes a rerun byte-identical: normalizing twice changes nothing."""
        once = normalize_text("  We   should\n\ngo ?  ")
        assert normalize_text(once) == once


class TestNothingIsEdited:
    @pytest.mark.parametrize(
        "text",
        [
            "we should go back to zephyrine",  # no capitalization added
            "We should go back to Zephyrine",  # no full stop added
            "Ah — no, wait, the, uh, the door",  # no disfluency removed
            "I seen him do it",  # no grammar corrected
            "It's 'the' one, so-called.",  # no quotes or hyphens rewritten
            "DM: roll for initiative",  # no speaker prefix interpreted
            "aaaaa",  # no invented words
        ],
    )
    def test_a_mangled_but_real_sentence_survives_verbatim(self, text: str) -> None:
        assert normalize_text(text) == text

    def test_punctuation_inside_a_word_is_untouched(self) -> None:
        assert normalize_text("don't stop") == "don't stop"

    def test_an_ellipsis_is_not_rewritten(self) -> None:
        assert normalize_text("well...") == "well..."


class TestTheComparisonKey:
    def test_it_ignores_case_and_punctuation(self) -> None:
        assert comparison_key("We should go!") == comparison_key("we should go")

    def test_it_is_never_what_gets_rendered(self) -> None:
        """A separate function precisely so nothing can accidentally render it."""
        assert normalize_text("We should go!") == "We should go!"
        assert comparison_key("We should go!") == "we should go"

    def test_punctuation_only_text_has_no_words(self) -> None:
        assert word_count("...") == 0
        assert comparison_key("...") == ""

    def test_words_are_counted_on_the_key(self) -> None:
        assert word_count("We should go, back!") == 4


class TestSimilarity:
    def test_identical_text_is_a_thousand(self) -> None:
        assert similarity_permille("We should go", "We should go") == 1000

    def test_case_and_punctuation_do_not_matter(self) -> None:
        assert similarity_permille("We should go!", "we should go") == 1000

    def test_it_compares_words_rather_than_characters(self) -> None:
        """Characters rate "no" against "now" at 800 per-mille — most of the way to a
        collapse threshold for two utterances that mean opposite things."""
        assert similarity_permille("no", "now") == 0

    def test_unrelated_text_scores_low(self) -> None:
        assert similarity_permille("We should go back", "Absolutely not") == 0

    def test_one_word_different_in_five_scores_high_but_not_perfect(self) -> None:
        score = similarity_permille(
            "we should go back to Zephyrine", "we should go back to Zephyrin"
        )
        assert 800 <= score < 1000

    def test_it_is_symmetric_enough_to_be_used_either_way_round(self) -> None:
        first = similarity_permille("we should go back now", "we should go back")
        second = similarity_permille("we should go back", "we should go back now")
        assert first == second

    def test_empty_against_empty_is_a_thousand_and_empty_against_text_is_zero(self) -> None:
        assert similarity_permille("", "") == 1000
        assert similarity_permille("", "hello") == 0

    def test_the_score_is_an_integer(self) -> None:
        """It is compared against a threshold and written into an artifact; a float would
        make both depend on binary rounding (INV-02)."""
        assert isinstance(similarity_permille("a b c", "a b d"), int)
