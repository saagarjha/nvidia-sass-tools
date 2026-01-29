import re
from dataclasses import dataclass, field

# =============================================================================
# Constants
# =============================================================================

# Native widths for float immediate types (type name -> bit width)
# Integer types use format_width as native width (asserted to equal bf_width at parse time)
FLOAT_NATIVE_WIDTHS = {
    'F16Imm': 16,
    'E8M7Imm': 16,  # BF16
    'E6M9Imm': 16,
    'F32Imm': 32,
    'F64Imm': 64,
}


@dataclass
class Enum:
    name: str
    cases: dict = field(default_factory=dict)  # name -> value
    reverse: dict = field(default_factory=dict)  # value -> name (last wins)


@dataclass
class Table:
    name: str
    entries: list = field(default_factory=list)  # list of (keys tuple, value)
    reverse: dict = field(default_factory=dict)  # value -> keys tuple (first wins)


@dataclass
class FormatElement:
    role: str = ""  # "", "prefix", or "suffix"
    modifiers: list = field(default_factory=list)  # [!], [~], [-]


@dataclass
class Operand(FormatElement):
    enum_name: str = ""
    default: int | None = None  # None = always show, int = hide if matches
    var_name: str = ""


@dataclass
class Immediate(FormatElement):
    imm_type: str = ""  # SImm, UImm, RSImm, F16Imm, F32Imm, F64Imm, BITSET
    width: int = 0
    default: int | None = None  # None = always show, int = hide if matches
    var_name: str = ""


@dataclass
class Literal(FormatElement):
    char: str = ""


@dataclass
class ConditionalLiteral(FormatElement):
    """A literal that's conditionally emitted based on a variable."""
    char: str = ""
    var_name: str = ""  # Variable to check (e.g., "Sb")
    condition: str = ""  # Condition suffix (e.g., "absolute" -> checks Sb@absolute)


@dataclass
class Opcode(FormatElement):
    pass


@dataclass
class Composite(FormatElement):
    children: list = field(default_factory=list)
    hidden: bool = False


@dataclass
class RawInstruction:
    cls: str
    format: str
    mnemonic: str
    opcode: int
    bitfields: list = field(default_factory=list)
    unused: str = ""
    format_alias: str = ""  # Only for alternate classes




class Instruction:
    """Parsed instruction (data holder for generator)."""

    def __init__(self, raw: RawInstruction, bitfields: list, variables: dict, enums: dict = None, implicit_variables: dict = None):
        self.raw = raw
        self.mnemonic = raw.mnemonic
        self.opcode = raw.opcode
        self.bitfields = bitfields
        self.variables = variables
        self.enums = enums or {}
        self.implicit_variables = implicit_variables or {}  # var_name -> constant value


@dataclass
class Bitfield:
    ranges: list  # list of (high, low) tuples
    constraint: int | set | range = None  # single value, set of values, or range
    variables: list = field(default_factory=list)  # list of Variable refs
    lookup: 'Lookup' = None  # table/function lookup

    @property
    def width(self) -> int:
        return sum(h - l + 1 for h, l in self.ranges)


@dataclass
class Variable:
    """Base variable - stored directly in bitfield."""
    name: str
    bitfield: Bitfield = None
    enum_name: str = ""  # enum type from format (only if actual enum)


@dataclass
class Comparison:
    """A single comparison: lhs == rhs, where both are evaluated to ints.

    lhs: Variable name (str) to look up at runtime, or literal int
    rhs: Integer value (resolved from enum or literal at parse time)
    """
    lhs: str | int
    rhs: int


@dataclass
class Conversion:
    """Parsed conversion expression for determining immediate type at decode time.

    cases: List of (conditions, result_type) where conditions is a list of
           Comparisons that are OR'd together. Evaluated in order.
    default: The fallback type if no condition matches.
    """
    cases: list  # list of (list[Comparison], str) tuples
    default: str


@dataclass
class ImmediateVariable(Variable):
    """Variable for immediate values with type-specific decoding.

    imm_type: Base type from format string (SImm, UImm, RSImm, F16Imm, F32Imm, F64Imm, etc.)
    conversion: Optional parsed Conversion that determines actual type at decode time.
                If set, evaluate to get the real imm_type (overrides the format's type).
    format_width: Width from format string (used for integer width assertions).
    """
    imm_type: str = ""
    conversion: Conversion = None
    format_width: int = 0


@dataclass
class TableLookupVariable(Variable):
    """Variable from a table lookup - decode by looking up in table."""
    table_name: str = ""
    arg_index: int = 0  # which arg position in the lookup


@dataclass
class ScaledVariable(Variable):
    """Variable with SCALE/MULTIPLY encoding."""
    scale: int = 1
    multiply: int = 1


@dataclass
class IdenticalVariable(Variable):
    """Variable that shares encoding with another in IDENTICAL(a, b)."""
    other_index: int = 0  # 0 means paired with index 1, 1 means paired with index 0


@dataclass
class Lookup:
    name: str  # function/table name
    args: list  # list of argument names



class Architecture:
    """Parsed GPU architecture specification.

    Attributes:
        enums: Dict of enum name -> Enum object
        tables: Dict of table name -> Table object
        instructions: List of Instruction objects (fully parsed with bitfields/variables)
        by_opcode: Dict of opcode -> list of Instructions
        by_mnemonic: Dict of mnemonic -> list of Instructions
    """

    def __init__(self, content: str):
        self._content = content
        self._lines = content.splitlines()
        self._statements = {}

        # Parse everything on init
        self._parse_statements()
        self.enums = self._parse_enums()
        self.tables = self._parse_tables(self.enums)
        raw_instructions = self._parse_instructions()

        # Build full Instruction objects with parsed bitfields/variables
        self.instructions = []
        for raw in raw_instructions:
            bitfields, variables, implicit_variables = self._parse_bitfields(raw, self.enums, self.tables)
            self.instructions.append(Instruction(raw, bitfields, variables, self.enums, implicit_variables))

        # Build indexes
        self.by_opcode = {}
        self.by_mnemonic = {}
        for instr in self.instructions:
            self.by_opcode.setdefault(instr.opcode, []).append(instr)
            self.by_mnemonic.setdefault(instr.mnemonic, []).append(instr)

    def _parse_statements(self):
        start_line = None
        end_line = None
        for i, line in enumerate(self._lines):
            if line.strip() == "REGISTERS":
                start_line = i + 1
            elif line.strip() == "OPERATION PROPERTIES":
                end_line = i
                break

        if start_line is None or end_line is None:
            raise ValueError("Could not find REGISTERS or OPERATION PROPERTIES section")

        section = "\n".join(self._lines[start_line:end_line])
        section = re.sub(r"^TABLES\s*$", "", section, flags=re.MULTILINE)

        for part in section.split(";"):
            part = part.strip()
            if not part:
                continue
            match = re.match(r"^([\w.]+)\s*(.*)", part, re.DOTALL)
            self._statements[match.group(1)] = match.group(2).strip()

    def _parse_enums(self) -> dict:
        """Parse all enums, resolving combined enums inline."""
        enum_statements = {k: v for k, v in self._statements.items() if "->" not in v}
        enums = {}
        for name in enum_statements:
            enums[name] = self._parse_enum(name, enums)
        return enums

    def _parse_tables(self, enums: dict) -> dict:
        """Parse all tables. Returns tables dict."""
        table_statements = {k: v for k, v in self._statements.items() if "->" in v}
        tables = {}
        for name in table_statements:
            tables[name] = self._parse_table(name, enums)
        return tables

    def _parse_enum(self, name: str, resolved: dict = None) -> Enum:
        body = self._statements[name]
        enum = Enum(name=name)
        current_value = 0

        # Combined enum: = Enum1 + Enum2 + ...
        if re.match(r"^=\s*[\w.]+(\s*\+\s*[\w.]+)*$", body):
            refs = [r.strip() for r in body[1:].split("+")]
            for ref in refs:
                enum.cases.update(resolved[ref].cases)
                enum.reverse.update(resolved[ref].reverse)
            return enum

        for case in self._split_by_comma(body):
            case = case.strip()
            if not case:
                continue

            # Parse: name, optional (range), optional *, optional =value
            m = re.match(
                r'^"?(?P<name>[\w.]+)"?\s*'
                r'(?:\((?P<r1>\d+)\.\.(?P<r2>\d+)\))?\s*'
                r'\*?\s*'
                r'(?:=\s*(?:\((?P<v1>\d+)\.\.(?P<v2>\d+)\)|(?P<val>\d+|0x[\da-fA-F]+|0b[01_]+)))?$',
                case
            )
            assert m, f"Failed to parse enum case: {case} in {name}"

            base_name = m.group("name")
            r1, r2 = m.group("r1"), m.group("r2")
            v1, v2 = m.group("v1"), m.group("v2")
            val = m.group("val")

            if r1 is not None:  # Range
                for i in range(int(r1), int(r2) + 1):
                    if v1 is not None:  # With value range
                        v = int(v1) + (i - int(r1))
                    else:  # Use counter
                        v = current_value
                        current_value += 1
                    case_name = f"{base_name}{i}"
                    enum.cases[case_name] = v
                    enum.reverse[v] = case_name  # last wins
                if v1 is not None:
                    current_value = int(v2) + 1
            elif val is not None:  # Single with value
                v = self._parse_int(val)
                enum.cases[base_name] = v
                enum.reverse[v] = base_name  # last wins
                current_value = v + 1
            else:  # Plain name
                enum.cases[base_name] = current_value
                enum.reverse[current_value] = base_name  # last wins
                current_value += 1

        return enum

    def _split_by_comma(self, s: str) -> list:
        parts, current, depth = [], [], 0
        for char in s:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append("".join(current))
                current = []
                continue
            current.append(char)
        if current:
            parts.append("".join(current))
        return parts

    def _tokenize(self, s: str) -> list:
        tokens = []
        i = 0
        while i < len(s):
            # Skip whitespace
            if s[i].isspace():
                i += 1
                continue
            # Quoted char like '&'
            if s[i] == "'":
                end = s.index("'", i + 1) + 1
                tokens.append(s[i:end])
                i = end
                continue
            # Token: identifier, number, etc.
            j = i
            while j < len(s) and not s[j].isspace() and s[j] != "'":
                j += 1
            if j > i:
                tokens.append(s[i:j])
            i = j
        return tokens

    def _resolve_enum_ref(self, token: str, enums: dict | None):
        """Resolve Enum@Case to its integer value."""
        if "@" in token and enums:
            enum_name, case_name = token.split("@", 1)
            case_name = case_name.strip('"')
            # Typos in original file: DC has "nodc" not "noDC", MS has "noms" not "noMS"
            case_name = {"noDC": "nodc", "noMS": "noms"}.get(case_name, case_name)
            return enums[enum_name].cases[case_name]
        return None

    def _parse_table(self, name: str, enums: dict | None = None) -> Table:
        body = self._statements[name]
        table = Table(name=name)

        parts = body.split("->")

        def resolve_key(token: str):
            if token.startswith("'") and token.endswith("'"):
                return token[1:-1]
            if token == "-":
                return None
            resolved = self._resolve_enum_ref(token, enums)
            if resolved is not None:
                return resolved
            return self._parse_int(token) if self._is_int(token) else token

        def resolve_value(token: str) -> int:
            resolved = self._resolve_enum_ref(token, enums)
            if resolved is not None:
                return resolved
            return self._parse_int(token)

        keys = tuple(resolve_key(t) for t in self._tokenize(parts[0]))

        for part in parts[1:-1]:
            tokens = self._tokenize(part)
            val = resolve_value(tokens[0])
            table.entries.append((keys, val))
            if val not in table.reverse:  # first wins
                table.reverse[val] = keys
            keys = tuple(resolve_key(t) for t in tokens[1:])

        tokens = self._tokenize(parts[-1])
        val = resolve_value(tokens[0])
        table.entries.append((keys, val))
        if val not in table.reverse:  # first wins
            table.reverse[val] = keys

        return table

    # Format token patterns
    _VAR_PATTERN = r'[A-Za-z_][\w.]*(?:\([^)]*\))?\*?(?::[\w.]+)?'
    _FMT_TOKEN = re.compile(rf'''
        (?P<ws>\s+) |
        (?P<modifier>\[\|\|\]|\[[!~-]\]) |
        (?P<hidden_open>\$\(\s*\{{) |
        (?P<hidden_close>\}}\s*\)\$) |
        (?P<bracket_open>\[) |
        (?P<bracket_close>\]) |
        (?P<joiner>','|\+) |
        (?P<suffix>/) |
        (?P<prefix>@) |
        (?P<opcode>Opcode\b) |
        (?P<variable>{_VAR_PATTERN}) |
        (?P<char>.)
    ''', re.VERBOSE)
    _VAR_PARSE = re.compile(r'([\w.]+)(?:\(([^)]*)\))?\*?(?::([\w.]+))?')

    def parse_format(self, fmt: str, pos: int = 0, end_chars: str = "", role: str = "") -> tuple[list, int]:
        """Parse format string recursively. Returns (elements, end_position)."""
        elements = []
        modifiers = []

        while pos < len(fmt):
            if fmt[pos] in end_chars:
                break

            m = self._FMT_TOKEN.match(fmt, pos)
            tok = m.lastgroup
            val = m.group()
            pos = m.end()

            if tok == 'ws':
                continue
            elif tok == 'modifier':
                modifiers.append(val[1:-1])  # Extract content from [x] or [||]
            elif tok == 'hidden_open':
                # Find matching } )$
                depth = 1
                start = pos
                while depth > 0:
                    inner_m = self._FMT_TOKEN.match(fmt, pos)
                    if inner_m.lastgroup == 'hidden_open':
                        depth += 1
                    elif inner_m.lastgroup == 'hidden_close':
                        depth -= 1
                    if depth > 0:
                        pos = inner_m.end()
                inner, _ = self.parse_format(fmt[start:pos], 0)
                elements.append(Composite(children=inner, hidden=True, role=role, modifiers=modifiers))
                modifiers = []
                pos = inner_m.end()
            elif tok == 'bracket_open':
                # Find matching ]
                depth = 1
                start = pos
                while depth > 0:
                    inner_m = self._FMT_TOKEN.match(fmt, pos)
                    if inner_m.lastgroup == 'bracket_open':
                        depth += 1
                    elif inner_m.lastgroup == 'bracket_close':
                        depth -= 1
                    if depth > 0:
                        pos = inner_m.end()
                inner, _ = self.parse_format(fmt[start:pos], 0)
                # Brackets go before/after the Composite as siblings
                elements.append(Literal(char='[', role="prefix"))
                elements.append(Composite(children=inner, role=role, modifiers=modifiers))
                elements.append(Literal(char=']', role="suffix"))
                modifiers = []
                pos = inner_m.end()
            elif tok == 'joiner':
                char = ', ' if val == "','" else '+'
                elements.append(Literal(char=char, role="prefix"))
            elif tok == 'suffix':
                suffix_elems, pos = self.parse_format(fmt, pos, end_chars + " /,'+[\n", role="suffix")
                elements.extend(suffix_elems)
            elif tok == 'prefix':
                # @ prefix becomes a prefix literal
                elements.append(Literal(char=val, role="prefix"))
            elif tok == 'opcode':
                elements.append(Opcode(role=role, modifiers=modifiers))
                modifiers = []
            elif tok == 'variable':
                var_m = self._VAR_PARSE.match(val)
                enum_name = var_m.group(1)
                raw_default = (var_m.group(2) or "").strip('"') or None  # None if empty
                var_name = var_m.group(3) or ""
                if enum_name in ('SImm', 'UImm', 'RSImm', 'F16Imm', 'F32Imm', 'F64Imm', 'BITSET'):
                    # Immediate type: parse width/default from spec like "17/0*" or "17/0x0*"
                    # No default means "always show", explicit default means "hide if matches"
                    parts = raw_default.rstrip('*').split('/')
                    width = int(parts[0]) if parts[0] else 0
                    default_val = int(parts[1], 0) if len(parts) > 1 else None
                    elements.append(Immediate(imm_type=enum_name, width=width, default=default_val, var_name=var_name, role=role, modifiers=modifiers))
                else:
                    default_val = self.enums[enum_name].cases[raw_default] if raw_default else None
                    elements.append(Operand(enum_name=enum_name, default=default_val, var_name=var_name, role=role, modifiers=modifiers))
                modifiers = []
            elif tok == 'char':
                if val not in '}*':
                    elements.append(Literal(char=val, role=role))

        # Post-process: convert modifiers to explicit elements
        elements = self._expand_modifiers(elements)
        # Post-process: insert space between bracket composites when followed by suffix
        elements = self._insert_bracket_spacing(elements)

        return elements, pos

    def _expand_modifiers(self, elements: list) -> list:
        """Convert modifier metadata on operands into explicit ConditionalLiteral elements.

        For ||, the closing | position depends on whether a {hidden} group follows:
        - If {hidden} group exists: | goes BEFORE it (suffix operands stay OUTSIDE ||)
        - If no {hidden} group: | goes after ALL suffixes (suffix operands stay INSIDE ||)
        """
        result = []
        pending_abs_suffix = None  # (var_name, condition) when we need closing |

        def is_new_content_group(elem):
            """Check if element ends the || scope and should trigger closing |."""
            # { marks end of || scope - suffixes after { are outside ||
            if isinstance(elem, Literal) and elem.char == '{':
                return True
            # Hidden Composite marks end of || scope
            if isinstance(elem, Composite) and elem.hidden:
                return True
            # Opcode means we're at the end or malformed
            if isinstance(elem, Opcode):
                return True
            # New main operand (not suffix) starts a new group
            if isinstance(elem, (Operand, Immediate)) and elem.role == '':
                return True
            # Literal prefix that's not '[' or '(' indicates new group (like ', ')
            if isinstance(elem, Literal) and elem.role == 'prefix' and elem.char not in ['[', '(']:
                return True
            # Suffix operands (like /HSEL) stay INSIDE || unless { was hit first
            return False

        for elem in elements:
            # Emit pending | suffix when we hit a new content group
            if pending_abs_suffix and is_new_content_group(elem):
                var_name, condition = pending_abs_suffix
                result.append(ConditionalLiteral(
                    char='|', var_name=var_name, condition=condition, role='suffix'
                ))
                pending_abs_suffix = None

            # Skip { markers (they only serve to trigger the | suffix above)
            if isinstance(elem, Literal) and elem.char == '{':
                continue

            # Expand modifiers on operands/immediates
            if isinstance(elem, (Operand, Immediate)) and elem.modifiers and elem.var_name:
                mod_map = {'!': 'not', '-': 'negate', '~': 'invert', '||': 'absolute'}

                # Insert prefix elements for each modifier
                for mod in elem.modifiers:
                    if mod in mod_map:
                        char = '|' if mod == '||' else mod
                        result.append(ConditionalLiteral(
                            char=char, var_name=elem.var_name,
                            condition=mod_map[mod], role='prefix'
                        ))

                result.append(elem)

                # For ||, remember to add closing | after all suffixes
                if '||' in elem.modifiers:
                    pending_abs_suffix = (elem.var_name, 'absolute')
            else:
                result.append(elem)

        # Emit trailing | suffix at end of list
        if pending_abs_suffix:
            var_name, condition = pending_abs_suffix
            result.append(ConditionalLiteral(
                char='|', var_name=var_name, condition=condition, role='suffix'
            ))

        return result

    def _insert_bracket_spacing(self, elements: list) -> list:
        """Insert space between consecutive bracket composites when followed by a suffix.

        nvdisasm outputs 'c[bank] [addr]' (with space) when the constant operand
        has a sub-element modifier like /HSEL, /EXTRACT, etc. Without such modifiers,
        it outputs 'c[bank][addr]' (no space).

        The rule: if a suffix Operand/Immediate follows two consecutive bracket groups,
        insert a space between them.
        """
        result = []

        for i, elem in enumerate(elements):
            # Check if this is a suffix Operand or Immediate (not ConditionalLiteral)
            if isinstance(elem, (Operand, Immediate)) and elem.role == "suffix":
                # Look back for pattern: ] [ Composite ]
                # We need at least 4 elements before: ] [ Composite ]
                if len(result) >= 4:
                    r = result
                    if (isinstance(r[-1], Literal) and r[-1].char == ']' and
                        isinstance(r[-2], Composite) and
                        isinstance(r[-3], Literal) and r[-3].char == '[' and
                        isinstance(r[-4], Literal) and r[-4].char == ']'):
                        # Insert space before the '[' at position -3
                        space = Literal(char=' ')
                        result.insert(-3, space)

            result.append(elem)

        return result

    def _is_int(self, s: str) -> bool:
        if s.startswith("0x") or s.startswith("0b"):
            return True
        return s.isdigit() or (s.startswith("-") and s[1:].isdigit())

    def _parse_int(self, s: str) -> int:
        if s.startswith("0b"):
            return int(s.replace("_", ""), 2)
        return int(s, 0)

    def _parse_instructions(self) -> list[RawInstruction]:
        instructions = []
        content = self._content

        # Parse unused bit patterns: name_unused 'XXX...XXX...'
        unused_patterns = {}
        for m in re.finditer(r"(\w+_unused)\s+'([X.]+)'", content):
            unused_patterns[m.group(1)] = m.group(2)

        # Find all CLASS blocks (not ALTERNATE CLASS)
        # Also handle case where CLASS appears after ; on same line
        class_starts = []
        for m in re.finditer(r'(?:^|;)(CLASS ")', content, re.MULTILINE):
            class_starts.append(m.start(1))
        class_starts.append(len(content))  # sentinel

        for i in range(len(class_starts) - 1):
            block = content[class_starts[i]:class_starts[i + 1]]
            instructions.append(self._parse_instruction_block(block, unused_patterns))

        # Parse ALTERNATE CLASS blocks (only those with FORMAT_ALIAS)
        alt_class_starts = []
        for m in re.finditer(r'(?:^|;)(ALTERNATE CLASS ")', content, re.MULTILINE):
            alt_class_starts.append(m.start(1))
        alt_class_starts.append(len(content))

        for i in range(len(alt_class_starts) - 1):
            block = content[alt_class_starts[i]:alt_class_starts[i + 1]]
            if "FORMAT_ALIAS" in block:
                instructions.append(self._parse_instruction_block(block, unused_patterns, alternate=True))

        return instructions

    def _parse_instruction_block(self, block: str, unused_patterns: dict, alternate: bool = False) -> RawInstruction:
        # Extract class name
        prefix = "ALTERNATE CLASS" if alternate else "CLASS"
        cls = re.match(rf'{prefix}\s+"([^"]+)"', block).group(1)

        # Extract FORMAT (from FORMAT PREDICATE to ;)
        fmt = re.search(r'FORMAT PREDICATE\s+(.*?);', block, re.DOTALL).group(1).strip()

        # Extract FORMAT_ALIAS (only for alternate classes)
        format_alias = ""
        if alternate:
            format_alias = re.search(r'FORMAT_ALIAS\s+(.*?);', block, re.DOTALL).group(1).strip()

        # Extract OPCODES section
        opcodes_text = re.search(r'OPCODES\s+(.*?)(?=ENCODING|$)', block, re.DOTALL).group(1)
        opcode_entries = re.findall(r'([\w.]+)\s*=\s*(0[bx][\da-fA-F_]+|\d+)\s*;', opcodes_text)
        shortest = min(opcode_entries, key=lambda x: len(x[0]))
        mnemonic = shortest[0]
        opcode = self._parse_int(shortest[1])

        # Extract ENCODING bitfields (BITS_* entries separated by ;)
        encoding_text = re.search(r'ENCODING\s*(.*?)(?=ALTERNATE CLASS |CLASS |$)', block, re.DOTALL).group(1)
        bitfields = []
        for part in encoding_text.split(';'):
            part = part.strip()
            if part.startswith('BITS_'):
                bitfields.append(part)

        # Look up unused bits pattern by class name
        unused = unused_patterns[cls + "_unused"]

        return RawInstruction(cls=cls, format=fmt, mnemonic=mnemonic, opcode=opcode, bitfields=bitfields, unused=unused, format_alias=format_alias)

    def _get_value_range(self, var_name: str, var_types: dict, enums: dict, width: int) -> set | range:
        """Get possible values for a variable based on its type."""
        enum_name, _ = var_types[var_name]
        if enum_name in enums:
            return self._compress_constraint(set(enums[enum_name].cases.values()))
        # Immediate types - use full width range
        return range(1 << width)

    def _compress_constraint(self, constraint):
        """Normalize constraint: single-element to int, dense set to range."""
        if isinstance(constraint, range):
            if len(constraint) == 1:
                return constraint.start
            return constraint
        if isinstance(constraint, set):
            if len(constraint) == 0:
                return constraint
            if len(constraint) == 1:
                return next(iter(constraint))
            min_v, max_v = min(constraint), max(constraint)
            if max_v - min_v + 1 == len(constraint):
                return range(min_v, max_v + 1)
            return constraint
        return constraint

    def _parse_conversion(self, conversion_str: str, enums: dict) -> Conversion:
        """Parse a conversion expression like convertFloatType(cond1, type1, ..., default).

        Returns a Conversion with cases as list of (comparisons, result_type).
        Each comparison has lhs (var name or int) and rhs (int, resolved from enum).
        """
        m = re.match(r'convertFloatType\((.+)\)$', conversion_str)
        if not m:
            raise ValueError(f"Unknown conversion format: {conversion_str}")

        # Split args - but be careful with nested commas (there aren't any currently)
        args = [a.strip() for a in m.group(1).split(',')]

        def parse_comparison(cond: str) -> Comparison:
            """Parse 'var == `ENUM@CASE' or '1==1' into Comparison with resolved int values."""
            cond = cond.strip()
            parts = cond.split('==')
            assert len(parts) == 2, f"Expected == comparison: {cond}"
            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()

            # Parse LHS - either variable name or literal int
            lhs = int(lhs_str) if lhs_str.isdigit() else lhs_str

            # Parse RHS - either `ENUM@CASE (resolve to int) or literal int
            if rhs_str.isdigit():
                rhs = int(rhs_str)
            else:
                enum_match = re.match(r'`(\w+)@([\w.]+)$', rhs_str)
                assert enum_match, f"Expected `ENUM@CASE or int: {rhs_str}"
                enum_name, case_name = enum_match.group(1), enum_match.group(2)
                rhs = enums[enum_name].cases[case_name]

            return Comparison(lhs, rhs)

        def parse_condition(cond_str: str) -> list:
            """Parse 'comp1 || comp2' into list of Comparisons."""
            parts = cond_str.split('||')
            return [parse_comparison(part) for part in parts]

        cases = []
        i = 0
        while i < len(args) - 1:
            cond_str = args[i]
            result_type = args[i + 1]
            comparisons = parse_condition(cond_str)
            cases.append((comparisons, result_type))
            i += 2

        default = args[-1] if len(args) % 2 == 1 else args[-1]
        return Conversion(cases=cases, default=default)

    def _parse_bitfield_lhs(self, lhs: str) -> Bitfield:
        """Parse a BITS_<width>_<high>_<low>... string into a Bitfield."""
        m = re.match(r'BITS_(\d+)_(.+)', lhs)
        width = int(m.group(1))
        rest = m.group(2).split('_')

        # Consume pairs until we have enough bits
        ranges = []
        bits_so_far = 0
        i = 0
        while bits_so_far < width and i + 1 < len(rest):
            high, low = int(rest[i]), int(rest[i + 1])
            ranges.append((high, low))
            bits_so_far += high - low + 1
            i += 2

        bf = Bitfield(ranges=ranges)
        assert bf.width == width, f"Width mismatch: declared {width}, got {bf.width}"
        return bf

    def _parse_bitfields(self, instr: RawInstruction, enums: dict, tables: dict = None) -> tuple[list[Bitfield], dict[str, Variable], dict[str, int]]:
        """Parse bitfields from a RawInstruction."""
        bitfields = []
        variables = {}

        # Parse format to build var_name -> (enum_name, width, default) mapping
        format_elements, _ = self.parse_format(instr.format)
        var_types = self._collect_var_types(format_elements, enums)

        for bf_str in instr.bitfields:
            parts = bf_str.split('=', 1)
            lhs = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ''

            # Determine constraint and variables based on value
            if value == 'Opcode':
                bf = self._parse_bitfield_lhs(lhs)
                bf.constraint = instr.opcode
            elif value.startswith('*') and value[1:].strip().isdigit():
                bf = self._parse_bitfield_lhs(lhs)
                bf.constraint = int(value[1:].strip())
            elif value.isdigit():
                bf = self._parse_bitfield_lhs(lhs)
                bf.constraint = int(value)
            elif value.startswith('0x') or value.startswith('0b'):
                bf = self._parse_bitfield_lhs(lhs)
                bf.constraint = self._parse_int(value)
            elif '(' in value and 'SCALE' not in value and 'MULTIPLY' not in value:
                # Unified lookup parsing: extract name(args), possibly with var prefix
                conv_m = re.match(r'\*?(\w+)\s+(convert\w+)\((.+)\)$', value, re.DOTALL)
                if conv_m:
                    # Conversion: "var_name convertTag(args)"
                    var_name, lookup_name, args_str = conv_m.group(1), conv_m.group(2), conv_m.group(3)
                    args = [a.strip() for a in args_str.split(',')]
                else:
                    m = re.match(r'\*?(\w+)\((.+)\)$', value, re.DOTALL)
                    assert m, f"Failed to parse lookup: {bf_str}"
                    lookup_name = m.group(1)
                    args = [a.strip() for a in m.group(2).split(',')]
                    var_name = None

                # Dispatch based on lookup_name
                if lookup_name.startswith('ConstBankAddress'):
                    # Multi-output: BITS_bank,BITS_offset = ConstBankAddress{0,2}(bank, addr)
                    lhs_parts = lhs.split(',')
                    assert len(lhs_parts) == 2 and len(args) == 2, f"ConstBankAddress expects 2 args: {bf_str}"

                    # Bank field (first) - always a simple variable
                    bank_bf = self._parse_bitfield_lhs(lhs_parts[0].strip())
                    bank_name = args[0]
                    bank_type, _ = var_types.get(bank_name, ('', 0))
                    bank_var = ImmediateVariable(name=bank_name, bitfield=bank_bf, imm_type=bank_type, format_width=bank_bf.width)
                    bank_bf.variables.append(bank_var)
                    bank_bf.constraint = range(1 << bank_bf.width)
                    bitfields.append(bank_bf)
                    variables[bank_name] = bank_var

                    # Offset field (second) - ConstBankAddress2 uses SCALE 4, ConstBankAddress0 doesn't
                    offset_bf = self._parse_bitfield_lhs(lhs_parts[1].strip())
                    offset_name = args[1]
                    offset_type, _ = var_types.get(offset_name, ('', 0))
                    if lookup_name == 'ConstBankAddress2':
                        offset_var = ScaledVariable(name=offset_name, bitfield=offset_bf, scale=4, multiply=1, enum_name=offset_type)
                    else:
                        offset_var = ImmediateVariable(name=offset_name, bitfield=offset_bf, imm_type=offset_type, format_width=offset_bf.width)
                    offset_bf.variables.append(offset_var)
                    offset_bf.constraint = range(1 << offset_bf.width)
                    bitfields.append(offset_bf)
                    variables[offset_name] = offset_var
                    continue
                elif lookup_name == 'IDENTICAL':
                    bf = self._parse_bitfield_lhs(lhs)
                    var1, var2 = args
                    width1 = self._get_var_width(var1, var_types)
                    width2 = self._get_var_width(var2, var_types)
                    assert width1 == width2 == bf.width, f"IDENTICAL width mismatch: {var1}={width1}, {var2}={width2}, bitfield={bf.width}"
                    bf.constraint = self._get_value_range(var1, var_types, enums, bf.width)
                    for i, arg in enumerate(args):
                        enum_name = var_types[arg][0]
                        var = IdenticalVariable(name=arg, bitfield=bf, enum_name=enum_name, other_index=1-i)
                        bf.variables.append(var)
                        variables[var.name] = var
                elif lookup_name.startswith('convert'):
                    bf = self._parse_bitfield_lhs(lhs)
                    conversion_str = f"{lookup_name}({','.join(args)})"
                    conversion = self._parse_conversion(conversion_str, enums)
                    # Get format type info - the format type is a placeholder, conversion determines actual type
                    fmt_type, fmt_width = var_types[var_name]
                    var = ImmediateVariable(
                        name=var_name, bitfield=bf, imm_type=fmt_type,
                        conversion=conversion, format_width=fmt_width
                    )
                    bf.variables.append(var)
                    bf.constraint = range(1 << bf.width)
                    variables[var.name] = var
                elif lookup_name in tables:
                    bf = self._parse_bitfield_lhs(lhs)
                    bf.lookup = Lookup(name=lookup_name, args=args)
                    table = tables[lookup_name]

                    # Filter table entries to only those where all enum outputs are valid
                    valid_encoded = set()
                    for keys, encoded in table.entries:
                        all_valid = True
                        for i, arg in enumerate(args):
                            decoded_val = keys[i]
                            enum_name, _ = var_types.get(arg, ('', 0))
                            if enum_name and enum_name in enums:
                                # Check if decoded value is in enum's valid values
                                if decoded_val not in enums[enum_name].cases.values():
                                    all_valid = False
                                    break
                        if all_valid:
                            valid_encoded.add(encoded)

                    bf.constraint = self._compress_constraint(valid_encoded)
                    for i, arg in enumerate(args):
                        var = TableLookupVariable(name=arg, bitfield=bf, table_name=lookup_name, arg_index=i)
                        bf.variables.append(var)
                        variables[var.name] = var
                else:
                    assert False, f"Unknown lookup: {lookup_name} in {bf_str}"
            elif 'SCALE' in value or 'MULTIPLY' in value:
                bf = self._parse_bitfield_lhs(lhs)
                # Variable with SCALE/MULTIPLY: varname SCALE N or varname MULTIPLY N SCALE M
                # Decoding: displayed = (encoded * scale) / multiply
                m = re.match(r'(\w+)(?:\s+MULTIPLY\s+(\d+))?(?:\s+SCALE\s+(\d+))?', value)
                assert m, f"Failed to parse SCALE/MULTIPLY: {bf_str}"
                var_name = m.group(1)
                multiply = int(m.group(2)) if m.group(2) else 1
                scale = int(m.group(3)) if m.group(3) else 1
                var = ScaledVariable(name=var_name, bitfield=bf, scale=scale, multiply=multiply)
                if var_name in var_types:
                    var.enum_name = var_types[var_name][0]
                bf.variables.append(var)
                # Constraint is the raw encoded range (ScaledVariable handles the transform)
                bf.constraint = range(1 << bf.width)
                variables[var.name] = var
            elif '@' in value:
                bf = self._parse_bitfield_lhs(lhs)
                # Modifier variable like Pg@not
                var = Variable(name=value, bitfield=bf)
                bf.variables.append(var)
                bf.constraint = range(2)  # boolean
                assert bf.width == 1, f"Modifier {value} should be 1 bit, got {bf.width}"
                variables[var.name] = var
            else:
                bf = self._parse_bitfield_lhs(lhs)
                # Variable reference (strip leading * if present)
                var_name = value.lstrip('*')
                var = Variable(name=var_name, bitfield=bf)
                bf.variables.append(var)
                type_name, _ = var_types[var_name]
                if type_name in enums:
                    var.enum_name = type_name
                    constraint = self._compress_constraint(set(enums[type_name].cases.values()))
                    bf.constraint = constraint
                    max_val = max(enums[type_name].cases.values())
                    assert max_val < (1 << bf.width), f"Enum {type_name} max {max_val} doesn't fit in {bf.width} bits"
                elif type_name in ('SImm', 'UImm', 'RSImm', 'BITSET') or type_name in FLOAT_NATIVE_WIDTHS:
                    # Replace with ImmediateVariable
                    _, format_width = var_types[var_name]
                    var = ImmediateVariable(
                        name=var_name, bitfield=bf, imm_type=type_name,
                        format_width=format_width
                    )
                    bf.variables = [var]
                    bf.constraint = range(1 << bf.width)

                    # Width assertions:
                    # - Integers: bitfield width must equal format width
                    # - Floats: bitfield width must not exceed native width
                    if type_name in ('SImm', 'UImm', 'RSImm', 'BITSET'):
                        assert bf.width == format_width, \
                            f"Integer {type_name} width mismatch: bf={bf.width}, format={format_width}"
                    else:
                        native_width = FLOAT_NATIVE_WIDTHS[type_name]
                        assert bf.width <= native_width, \
                            f"Float {type_name} bf width {bf.width} > native {native_width}"
                else:
                    assert False, f"Unknown type {type_name} for variable {var_name}"
                variables[var.name] = var

            bitfields.append(bf)

        # Parse unused bits pattern and create unconstrained bitfields
        # Pattern is 128 chars, bit 127 at index 0, bit 0 at index 127
        # 'X' = unused bit, '.' = used bit
        unused = instr.unused
        i = 0
        while i < len(unused):
            if unused[i] == 'X':
                # Find contiguous run of X's
                start = i
                while i < len(unused) and unused[i] == 'X':
                    i += 1
                # Convert to bit positions (127 - index)
                high = 127 - start
                low = 127 - (i - 1)
                bf = Bitfield(ranges=[(high, low)], constraint=None)
                bitfields.append(bf)
            else:
                i += 1

        # Verify bitfields cover the full 128-bit instruction
        total_bits = sum(bf.width for bf in bitfields)
        assert total_bits == 128, f"Bitfields cover {total_bits} bits, expected 128 for {instr.mnemonic} ({instr.cls})"

        # Normalize all constraints
        for bf in bitfields:
            bf.constraint = self._compress_constraint(bf.constraint)

        # Find implicit variables: in format but not in bitfields
        implicit_variables = {}
        for var_name, (enum_name, width) in var_types.items():
            if var_name not in variables:
                values = set(enums[enum_name].cases.values())
                assert len(values) == 1, \
                    f"Implicit variable {var_name} (enum {enum_name}) must have exactly 1 value, got {values}"
                implicit_variables[var_name] = next(iter(values))

        return bitfields, variables, implicit_variables

    def _collect_var_types(self, elements: list, enums: dict, result: dict = None) -> dict:
        """Walk format elements and collect var_name -> (enum_name, width) mapping."""
        if result is None:
            result = {}
        for elem in elements:
            if isinstance(elem, Operand) and elem.var_name:
                enum_name = elem.enum_name
                raw_default = elem.default
                # Parse width from enum or immediate spec
                if enum_name in enums:
                    width = max(enums[enum_name].cases.values()).bit_length()
                elif raw_default and raw_default.split('/')[0].isdigit():
                    width = int(raw_default.split('/')[0])
                else:
                    width = 0
                result[elem.var_name] = (enum_name, width)
            if isinstance(elem, Immediate) and elem.var_name:
                result[elem.var_name] = (elem.imm_type, elem.width)
            if isinstance(elem, Composite):
                self._collect_var_types(elem.children, enums, result)
        return result

    def _get_var_width(self, var_name: str, var_types: dict) -> int:
        """Get the bit width of a variable from var_types."""
        return var_types[var_name][1]


