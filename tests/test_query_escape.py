"""Escape layer: the injection firewall. All user strings go through here."""
from datetime import date, datetime

import pytest

from graphstore.query.escape import dsl_identifier, dsl_literal, dsl_node_id


class TestDslLiteral:
    def test_str_plain(self):
        assert dsl_literal("hello") == '"hello"'

    def test_str_with_double_quotes(self):
        assert dsl_literal('a"b') == r'"a\"b"'

    def test_str_with_backslash(self):
        assert dsl_literal(r"a\b") == r'"a\\b"'

    def test_str_injection_attempt(self):
        """Classic injection vector: user string tries to break out."""
        malicious = 'foo"; DROP ALL; --'
        out = dsl_literal(malicious)
        assert out.startswith('"')
        assert out.endswith('"')
        # embedded quote is escaped
        assert r'\"' in out
        # the dangerous close-quote + semicolon is now a literal inside a string
        assert "DROP ALL" in out  # still there but as content, not syntax

    def test_int(self):
        assert dsl_literal(42) == "42"

    def test_int_zero(self):
        assert dsl_literal(0) == "0"

    def test_int_negative(self):
        assert dsl_literal(-7) == "-7"

    def test_float(self):
        assert dsl_literal(0.5) == "0.5"

    def test_bool_true(self):
        assert dsl_literal(True) == "true"

    def test_bool_false(self):
        assert dsl_literal(False) == "false"

    def test_none(self):
        assert dsl_literal(None) == "null"

    def test_list_of_strings(self):
        assert dsl_literal(["a", "b"]) == '("a", "b")'

    def test_list_of_ints(self):
        assert dsl_literal([1, 2, 3]) == "(1, 2, 3)"

    def test_list_mixed(self):
        assert dsl_literal(["a", 1, True]) == '("a", 1, true)'

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            dsl_literal([])

    def test_tuple(self):
        assert dsl_literal(("a", "b")) == '("a", "b")'

    def test_date(self):
        assert dsl_literal(date(2024, 3, 15)) == '"2024-03-15"'

    def test_datetime(self):
        assert dsl_literal(datetime(2024, 3, 15, 10, 30, 0)) == '"2024-03-15T10:30:00"'

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="unsupported DSL value type"):
            dsl_literal(object())

    def test_bool_before_int(self):
        """bool must be checked before int since bool is a subclass of int."""
        # If int check ran first, True -> "1". Verify we emit "true".
        assert dsl_literal(True) == "true"
        assert dsl_literal(False) == "false"


class TestDslIdentifier:
    def test_valid(self):
        assert dsl_identifier("kind") == "kind"
        assert dsl_identifier("my_field_1") == "my_field_1"
        assert dsl_identifier("__event_at__") == "__event_at__"

    def test_empty(self):
        with pytest.raises(ValueError):
            dsl_identifier("")

    def test_starts_with_digit(self):
        with pytest.raises(ValueError):
            dsl_identifier("1field")

    def test_with_dash(self):
        with pytest.raises(ValueError):
            dsl_identifier("my-field")

    def test_with_space(self):
        with pytest.raises(ValueError):
            dsl_identifier("my field")

    def test_injection_attempt(self):
        with pytest.raises(ValueError):
            dsl_identifier('kind"; DROP ALL; --')


class TestDslNodeId:
    def test_valid(self):
        assert dsl_node_id("mem:1") == '"mem:1"'

    def test_with_special_chars_escaped(self):
        # IDs can contain colons, dashes - but quotes must be escaped
        assert dsl_node_id("ent:paris-eiffel") == '"ent:paris-eiffel"'

    def test_injection_attempt(self):
        malicious = 'mem:1"; DROP ALL; --'
        out = dsl_node_id(malicious)
        assert out.startswith('"')
        assert out.endswith('"')
        assert r'\"' in out

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            dsl_node_id("")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            dsl_node_id(42)
