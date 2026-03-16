"""TDL (Type Description Language) parser for HPSG grammars.

Recursive descent parser that reads TDL source files (e.g., the English
Resource Grammar) and produces an AST of type definitions, feature
structures, include directives, and section markers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# === AST nodes ===


@dataclass
class TdlToken:
    kind: str  # "ident", "string", "op", "directive", "docstring"
    value: str
    line: int
    col: int


@dataclass
class TdlFeature:
    """Base class for feature values in a feature structure."""
    pass


@dataclass
class TdlType(TdlFeature):
    name: str


@dataclass
class TdlString(TdlFeature):
    value: str


@dataclass
class TdlCoref(TdlFeature):
    name: str


@dataclass
class TdlList(TdlFeature):
    elements: list[TdlFeature]
    open: bool  # True if ends with ...


@dataclass
class TdlFeatStruct(TdlFeature):
    features: dict[str, TdlFeature]


@dataclass
class TdlConj(TdlFeature):
    parts: list[TdlFeature]


@dataclass
class TdlDefinition:
    name: str
    supertypes: list[str]
    body: TdlFeature | None
    docstring: str | None
    section: str | None
    is_instance: bool
    is_addendum: bool
    suffix: str | None


@dataclass
class TdlInclude:
    filename: str


@dataclass
class TdlDirective:
    kind: str  # "begin", "end", "letter-set"
    content: str


# === Tokenizer ===


def tokenize_tdl(text: str) -> list[TdlToken]:
    """Split TDL text into tokens, stripping comments."""
    tokens: list[TdlToken] = []
    i = 0
    line = 1
    col = 1

    while i < len(text):
        c = text[i]

        # Line comment
        if c == ';':
            while i < len(text) and text[i] != '\n':
                i += 1
            continue

        # Block comment #| ... |#
        if c == '#' and i + 1 < len(text) and text[i + 1] == '|':
            i += 2
            col += 2
            depth = 1
            while i < len(text) and depth > 0:
                if text[i] == '#' and i + 1 < len(text) and text[i + 1] == '|':
                    depth += 1
                    i += 2
                    col += 2
                elif text[i] == '|' and i + 1 < len(text) and text[i + 1] == '#':
                    depth -= 1
                    i += 2
                    col += 2
                else:
                    if text[i] == '\n':
                        line += 1
                        col = 1
                    else:
                        col += 1
                    i += 1
            continue

        # Whitespace
        if c in ' \t\r':
            i += 1
            col += 1
            continue

        if c == '\n':
            i += 1
            line += 1
            col = 1
            continue

        # Docstring """..."""
        if c == '"' and i + 2 < len(text) and text[i + 1] == '"' and text[i + 2] == '"':
            start_line = line
            start_col = col
            i += 3
            col += 3
            doc_start = i
            while i < len(text):
                if text[i] == '"' and i + 2 < len(text) and text[i + 1] == '"' and text[i + 2] == '"':
                    doc_text = text[doc_start:i]
                    tokens.append(TdlToken("docstring", doc_text, start_line, start_col))
                    i += 3
                    col += 3
                    break
                if text[i] == '\n':
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
            continue

        # String "..."
        if c == '"':
            start_line = line
            start_col = col
            i += 1
            col += 1
            str_start = i
            while i < len(text) and text[i] != '"':
                if text[i] == '\\':
                    i += 1
                    col += 1
                if text[i] == '\n':
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
            str_val = text[str_start:i]
            tokens.append(TdlToken("string", str_val, start_line, start_col))
            if i < len(text):
                i += 1  # skip closing "
                col += 1
            continue

        # %(letter-set ...) or %suffix at file scope
        # We detect % at start of what could be a directive
        if c == '%':
            start_line = line
            start_col = col
            # Check for %(letter-set
            if i + 1 < len(text) and text[i + 1] == '(':
                # %(letter-set (...))
                paren_depth = 0
                j = i
                while j < len(text):
                    if text[j] == '(':
                        paren_depth += 1
                    elif text[j] == ')':
                        paren_depth -= 1
                        if paren_depth == 0:
                            j += 1
                            break
                    j += 1
                content = text[i:j]
                tokens.append(TdlToken("directive", content, start_line, start_col))
                # Update line/col
                for ch in text[i:j]:
                    if ch == '\n':
                        line += 1
                        col = 1
                    else:
                        col += 1
                i = j
                continue
            # %suffix - but could be embedded in definition body
            if i + 6 < len(text) and text[i + 1:i + 7] == 'suffix':
                # Collect everything until next line starting with non-whitespace
                # that isn't a suffix continuation, or until we hit a docstring or ident
                j = i + 7
                col += 7
                # Collect suffix content until we hit a newline followed by
                # a non-suffix line (docstring, type name, etc.)
                # Simpler: collect until end of all (pattern pattern) groups
                suffix_content = ""
                while j < len(text) and text[j] in ' \t':
                    j += 1
                suffix_start = j
                # Collect all (group) entries
                while j < len(text):
                    if text[j] == '(':
                        paren_depth = 1
                        j += 1
                        while j < len(text) and paren_depth > 0:
                            if text[j] == '(':
                                paren_depth += 1
                            elif text[j] == ')':
                                paren_depth -= 1
                            j += 1
                        # Skip whitespace between groups
                        while j < len(text) and text[j] in ' \t\n\r':
                            if text[j] == '\n':
                                line += 1
                                col = 1
                            else:
                                col += 1
                            j += 1
                    else:
                        break
                suffix_content = text[suffix_start:j].strip()
                tokens.append(TdlToken("suffix", suffix_content, start_line, start_col))
                # Update line tracking for consumed text
                for ch in text[i:suffix_start]:
                    if ch == '\n':
                        line += 1
                        col = 1
                    else:
                        col += 1
                i = j
                continue
            # Unknown % directive -- skip to end of line
            while i < len(text) and text[i] != '\n':
                i += 1
                col += 1
            continue

        # Operators: :=, :+, :begin, :end, :include, :status, :type, :instance
        if c == ':':
            start_line = line
            start_col = col
            # Check for := or :+
            if i + 1 < len(text) and text[i + 1] == '=':
                tokens.append(TdlToken("op", ":=", start_line, start_col))
                i += 2
                col += 2
                continue
            if i + 1 < len(text) and text[i + 1] == '+':
                tokens.append(TdlToken("op", ":+", start_line, start_col))
                i += 2
                col += 2
                continue
            # Keyword directive like :begin, :end, :include, :type, :instance, :status
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] in '-_'):
                j += 1
            keyword = text[i:j]
            tokens.append(TdlToken("op", keyword, start_line, start_col))
            col += j - i
            i = j
            continue

        # Ellipsis ... (must come before single-char '.' check)
        if c == '.' and i + 2 < len(text) and text[i + 1] == '.' and text[i + 2] == '.':
            tokens.append(TdlToken("op", "...", line, col))
            i += 3
            col += 3
            continue

        # Single-char operators
        if c in '&.[]<>,':
            tokens.append(TdlToken("op", c, line, col))
            i += 1
            col += 1
            continue

        # Hash (coreference prefix)
        if c == '#':
            tokens.append(TdlToken("op", "#", line, col))
            i += 1
            col += 1
            continue

        # Identifiers (type names, feature names)
        # TDL identifiers can contain letters, digits, -, _, *, +, and more
        if c.isalnum() or c in '_*!+-':
            start_line = line
            start_col = col
            j = i
            while j < len(text) and (
                text[j].isalnum()
                or text[j] in '_-*+/\'.'
            ):
                # Don't consume '.' at end (statement terminator)
                # unless it's part of an identifier like "a.b"
                if text[j] == '.':
                    # Check if next char continues the identifier
                    if j + 1 < len(text) and (
                        text[j + 1].isalnum()
                        or text[j + 1] in '_-*+/'
                    ):
                        j += 1
                    else:
                        break
                else:
                    j += 1
            ident = text[i:j]
            tokens.append(TdlToken("ident", ident, start_line, start_col))
            col += j - i
            i = j
            continue

        # Skip unknown characters
        i += 1
        col += 1

    return tokens


# === Parser ===


class _Parser:
    """Recursive descent parser for TDL token streams."""

    def __init__(self, tokens: list[TdlToken]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._section_context: str | None = None
        self._is_instance: bool = False

    def _peek(self) -> TdlToken | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> TdlToken:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> TdlToken:
        tok = self._peek()
        if tok is None:
            raise ValueError(f"Unexpected end of input, expected {kind} {value}")
        if tok.kind != kind or (value is not None and tok.value != value):
            raise ValueError(
                f"Expected {kind} {value!r} at line {tok.line}:{tok.col}, "
                f"got {tok.kind} {tok.value!r}"
            )
        return self._advance()

    def _at(self, kind: str, value: str | None = None) -> bool:
        tok = self._peek()
        if tok is None:
            return False
        if tok.kind != kind:
            return False
        if value is not None and tok.value != value:
            return False
        return True

    def parse(self) -> list[TdlDefinition | TdlInclude | TdlDirective]:
        results: list[TdlDefinition | TdlInclude | TdlDirective] = []
        while self._pos < len(self._tokens):
            tok = self._peek()
            if tok is None:
                break

            # Letter-set / suffix directives at file scope
            if tok.kind == "directive":
                self._advance()
                if "letter-set" in tok.value:
                    results.append(TdlDirective("letter-set", tok.value))
                else:
                    results.append(TdlDirective("directive", tok.value))
                continue

            # Section directives
            if tok.kind == "op" and tok.value == ":begin":
                directive = self._parse_begin_section()
                if directive is not None:
                    results.append(directive)
                continue

            if tok.kind == "op" and tok.value == ":end":
                directive = self._parse_end_section()
                if directive is not None:
                    results.append(directive)
                continue

            # Include directive
            if tok.kind == "op" and tok.value == ":include":
                self._advance()
                filename_tok = self._expect("string")
                # Consume trailing .
                if self._at("op", "."):
                    self._advance()
                results.append(TdlInclude(filename_tok.value))
                continue

            # Type/instance definition
            if tok.kind == "ident":
                defn = self._parse_definition()
                if defn is not None:
                    results.append(defn)
                continue

            # Skip unexpected tokens
            self._advance()

        return results

    def _parse_begin_section(self) -> TdlDirective | None:
        self._advance()  # :begin
        # Collect section keywords until .
        section_parts: list[str] = []
        status: str | None = None
        while not self._at("op", "."):
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "op" and tok.value.startswith(":"):
                keyword = tok.value[1:]  # strip leading :
                section_parts.append(keyword)
                self._advance()
                # If :status, next token is the status value
                if keyword == "status":
                    if self._peek() and self._peek().kind == "ident":
                        status = self._advance().value
                continue
            self._advance()
        # Consume .
        if self._at("op", "."):
            self._advance()

        if "instance" in section_parts:
            self._is_instance = True
            if status:
                self._section_context = f"instance:{status}"
            else:
                self._section_context = "instance"
        elif "type" in section_parts:
            self._section_context = "type"

        content = ":".join(section_parts)
        if status:
            content += f":{status}"
        return TdlDirective("begin", content)

    def _parse_end_section(self) -> TdlDirective | None:
        self._advance()  # :end
        # Collect what we're ending
        section_parts: list[str] = []
        while not self._at("op", "."):
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "op" and tok.value.startswith(":"):
                section_parts.append(tok.value[1:])
            self._advance()
        if self._at("op", "."):
            self._advance()
        self._is_instance = False
        self._section_context = None
        return TdlDirective("end", ":".join(section_parts))

    def _parse_definition(self) -> TdlDefinition | None:
        name_tok = self._advance()
        name = name_tok.value

        # Expect := or :+
        tok = self._peek()
        if tok is None:
            return None
        if tok.kind != "op" or tok.value not in (":=", ":+"):
            # Not a definition -- might be a stray identifier. Skip to next .
            self._skip_to_dot()
            return None

        is_addendum = tok.value == ":+"
        self._advance()  # consume := or :+

        # Check for %suffix embedded in definition body
        suffix: str | None = None
        if self._at("suffix"):
            suffix_tok = self._advance()
            suffix = suffix_tok.value

        # Check for docstring
        docstring: str | None = None
        if self._at("docstring"):
            docstring = self._advance().value

        # Parse body: supertypes & feature structures
        supertypes: list[str] = []
        body: TdlFeature | None = None

        if is_addendum:
            # Addenda have no supertypes, just a feature structure
            body = self._parse_conjunction()
        else:
            # Parse supertypes and optional body
            body = self._parse_conjunction()
            # Extract supertypes from conjunction
            if isinstance(body, TdlType):
                supertypes = [body.name]
                body = None
            elif isinstance(body, TdlConj):
                # Separate type names from feature structures
                new_parts: list[TdlFeature] = []
                for part in body.parts:
                    if isinstance(part, TdlType):
                        supertypes.append(part.name)
                    else:
                        new_parts.append(part)
                if len(new_parts) == 0:
                    body = None
                elif len(new_parts) == 1:
                    body = new_parts[0]
                else:
                    body = TdlConj(new_parts)

        # Expect terminating .
        if self._at("op", "."):
            self._advance()

        return TdlDefinition(
            name=name,
            supertypes=supertypes,
            body=body,
            docstring=docstring,
            section=self._section_context,
            is_instance=self._is_instance,
            is_addendum=is_addendum,
            suffix=suffix,
        )

    def _parse_conjunction(self) -> TdlFeature | None:
        """Parse a conjunction of types and feature structures."""
        parts: list[TdlFeature] = []

        part = self._parse_term()
        if part is not None:
            parts.append(part)

        while self._at("op", "&"):
            self._advance()  # consume &
            # Check for docstring between & and next term
            if self._at("docstring"):
                # Skip inline docstring
                self._advance()
            part = self._parse_term()
            if part is not None:
                parts.append(part)

        if len(parts) == 0:
            return None
        if len(parts) == 1:
            return parts[0]
        return TdlConj(parts)

    def _parse_term(self) -> TdlFeature | None:
        """Parse a single term: type name, feature structure, string, coref, or list."""
        tok = self._peek()
        if tok is None:
            return None

        # Feature structure
        if tok.kind == "op" and tok.value == "[":
            return self._parse_feat_struct()

        # String
        if tok.kind == "string":
            self._advance()
            return TdlString(tok.value)

        # Coreference
        if tok.kind == "op" and tok.value == "#":
            self._advance()
            name_tok = self._peek()
            if name_tok and name_tok.kind == "ident":
                self._advance()
                return TdlCoref(name_tok.value)
            return TdlCoref("")

        # List
        if tok.kind == "op" and tok.value == "<":
            return self._parse_list()

        # Docstring (skip and try again)
        if tok.kind == "docstring":
            self._advance()
            return self._parse_term()

        # Type name (identifier)
        if tok.kind == "ident":
            self._advance()
            return TdlType(tok.value)

        return None

    def _parse_feat_struct(self) -> TdlFeatStruct:
        """Parse [ FEAT val, FEAT2 val2 ]."""
        self._expect("op", "[")
        features: dict[str, TdlFeature] = {}

        while not self._at("op", "]"):
            tok = self._peek()
            if tok is None:
                break

            # Safety: bail on statement terminators inside feat struct
            if tok.kind == "op" and tok.value == ".":
                break

            # Feature name
            if tok.kind != "ident":
                # Skip unexpected token
                self._advance()
                continue

            feat_name = self._advance().value

            # Check for dot path
            parts = feat_name.split(".")
            if len(parts) > 1:
                # Expand dot path: F1.F2.F3 val -> F1 [ F2 [ F3 val ] ]
                val = self._parse_feature_value()
                # Build nested structure from inside out
                for part in reversed(parts[1:]):
                    val = TdlFeatStruct({part: val})
                features[parts[0]] = val
            else:
                val = self._parse_feature_value()
                features[feat_name] = val

            # Optional comma between features
            if self._at("op", ","):
                self._advance()

        if self._at("op", "]"):
            self._advance()

        return TdlFeatStruct(features)

    def _parse_feature_value(self) -> TdlFeature:
        """Parse the value part of a feature assignment."""
        # Could be a conjunction (type & [ ... ])
        val = self._parse_conjunction()
        if val is not None:
            return val
        return TdlType("")

    def _parse_list(self) -> TdlList:
        """Parse < elem1, elem2, ... >."""
        self._expect("op", "<")
        elements: list[TdlFeature] = []
        is_open = False

        while not self._at("op", ">"):
            tok = self._peek()
            if tok is None:
                break

            # Safety: bail on statement terminators inside list
            if tok.kind == "op" and tok.value == ".":
                break

            # Ellipsis
            if tok.kind == "op" and tok.value == "...":
                self._advance()
                is_open = True
                if self._at("op", ","):
                    self._advance()
                continue

            # Check for ! at end (diff-list notation in ERG)
            if tok.kind == "op" and tok.value == "!":
                self._advance()
                is_open = True
                continue

            start_pos = self._pos
            elem = self._parse_conjunction()
            if elem is not None:
                elements.append(elem)
            elif self._pos == start_pos:
                # No progress -- skip unexpected token to avoid infinite loop
                self._advance()

            if self._at("op", ","):
                self._advance()

        if self._at("op", ">"):
            self._advance()

        return TdlList(elements, is_open)

    def _skip_to_dot(self) -> None:
        """Skip tokens until the next statement-terminating dot."""
        depth = 0
        while self._pos < len(self._tokens):
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "op" and tok.value == "[":
                depth += 1
            elif tok.kind == "op" and tok.value == "]":
                depth -= 1
            elif tok.kind == "op" and tok.value == "." and depth == 0:
                self._advance()
                return
            self._advance()


def parse_tdl(tokens: list[TdlToken]) -> list[TdlDefinition | TdlInclude | TdlDirective]:
    """Parse a token stream into TDL AST nodes."""
    parser = _Parser(tokens)
    return parser.parse()


def parse_tdl_file(path: Path) -> list[TdlDefinition | TdlInclude | TdlDirective]:
    """Parse a single TDL file into AST nodes."""
    text = path.read_text(encoding="utf-8")
    tokens = tokenize_tdl(text)
    return parse_tdl(tokens)


def resolve_include(base_dir: Path, filename: str) -> Path:
    """Resolve a TDL :include directive to an absolute file path.

    Tries base_dir/filename first, then base_dir/filename.tdl.
    Raises FileNotFoundError with context if neither exists.
    """
    # Try exact path first
    candidate = base_dir / filename
    if candidate.exists():
        return candidate

    # Try with .tdl extension
    candidate_tdl = base_dir / (filename + ".tdl")
    if candidate_tdl.exists():
        return candidate_tdl

    raise FileNotFoundError(
        f"Cannot resolve include '{filename}' from {base_dir}: "
        f"neither {candidate} nor {candidate_tdl} exists"
    )


def parse_tdl_directory(
    top_file: Path,
    section_context: str | None = None,
) -> list[TdlDefinition | TdlDirective]:
    """Recursively parse a TDL grammar starting from a top-level file.

    Follows :include directives, resolves paths, propagates section
    context into included files. Returns all definitions from all files.
    """
    seen: set[Path] = set()
    results: list[TdlDefinition | TdlDirective] = []
    _parse_recursive(top_file, section_context, seen, results)
    return results


def _parse_recursive(
    file_path: Path,
    section_context: str | None,
    seen: set[Path],
    results: list[TdlDefinition | TdlDirective],
) -> None:
    """Recursively parse a TDL file and its includes."""
    resolved = file_path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)

    if not resolved.exists():
        logger.warning("TDL file not found: %s", resolved)
        return

    text = resolved.read_text(encoding="utf-8")
    tokens = tokenize_tdl(text)
    parser = _Parser(tokens)

    # If we have an inherited section context, apply it
    if section_context is not None:
        parser._section_context = section_context
        if section_context.startswith("instance"):
            parser._is_instance = True

    items = parser.parse()
    base_dir = resolved.parent

    # Track section state for include resolution
    current_section = section_context
    current_is_instance = section_context is not None and section_context.startswith("instance")

    for item in items:
        if isinstance(item, TdlDirective):
            if item.kind == "begin":
                # Parse section from content
                if "instance" in item.content:
                    current_is_instance = True
                    # Extract status if present
                    parts = item.content.split(":")
                    current_section = "instance"
                    for i, p in enumerate(parts):
                        if "status" in p and i + 1 < len(parts):
                            current_section = f"instance:{parts[i + 1].strip()}"
                elif "type" in item.content:
                    current_section = "type"
                    current_is_instance = False
            elif item.kind == "end":
                current_section = section_context  # revert to inherited
                current_is_instance = section_context is not None and section_context.startswith("instance")
            results.append(item)
        elif isinstance(item, TdlInclude):
            try:
                include_path = resolve_include(base_dir, item.filename)
                _parse_recursive(include_path, current_section, seen, results)
            except FileNotFoundError:
                logger.warning(
                    "Include not found: '%s' from %s", item.filename, file_path
                )
        elif isinstance(item, TdlDefinition):
            # Apply current section context if not already set
            if item.section is None and current_section is not None:
                # Mutate the definition to add section context
                item.section = current_section
                item.is_instance = current_is_instance
            results.append(item)
