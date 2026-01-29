#!/usr/bin/env python3
"""
Generate a standalone disassembler from an Architecture spec.

Usage:
    python generate_disassembler.py <spec_file> [output_file]
    python generate_disassembler.py spec.txt disasm.py
"""

import sys
from architecture import Architecture, FLOAT_NATIVE_WIDTHS


def _constraints_overlap(c1, c2):
    """Check if two constraints have any overlapping values."""
    if c1 is None or c2 is None:
        return True
    if isinstance(c1, int) and isinstance(c2, int):
        return c1 == c2
    if isinstance(c1, int):
        return c1 in c2
    if isinstance(c2, int):
        return c2 in c1
    if isinstance(c1, range) and isinstance(c2, range):
        return max(c1.start, c2.start) < min(c1.stop, c2.stop)
    if isinstance(c1, set) and isinstance(c2, set):
        return bool(c1 & c2)
    # range vs set
    r, s = (c1, c2) if isinstance(c1, range) else (c2, c1)
    return any(v in r for v in s)


def assert_instructions_distinguishable(instrs: list):
    """Assert all instruction variants in a group are distinguishable by their bitfield constraints."""
    for i, instr1 in enumerate(instrs):
        if '_pseudo' in instr1.raw.cls:
            continue
        for instr2 in instrs[i+1:]:
            if '_pseudo' in instr2.raw.cls:
                continue
            bf_map1 = {tuple(bf.ranges): bf for bf in instr1.bitfields}
            bf_map2 = {tuple(bf.ranges): bf for bf in instr2.bitfields}
            common = set(bf_map1) & set(bf_map2)

            assert any(
                not _constraints_overlap(bf_map1[k].constraint, bf_map2[k].constraint)
                for k in common
            ), f"Not distinguishable: {instr1.raw.cls} vs {instr2.raw.cls}"


def make_class_name(cls: str) -> str:
    """Convert instruction class name to valid Python identifier."""
    name = cls.replace('.', '_').replace('-', '_')
    if name[0].isdigit():
        name = '_' + name
    return name


def sanitize_var_name(name: str) -> str:
    """Convert variable name to valid Python identifier."""
    name = name.replace('@', '_').replace('.', '_').replace('-', '_')
    if name[0].isdigit():
        name = '_' + name
    return name


def get_prefix_info(prefixes):
    """Extract joiner, literal, and conditional prefix info from prefix elements."""
    from architecture import Literal, ConditionalLiteral
    joiners = []
    literals = []
    cond_prefixes = []  # (char, var_name, condition)
    for p in prefixes:
        if isinstance(p, Literal):
            if p.char in [', ', '+']:
                joiners.append(p.char)
            else:
                resets = p.char == '['
                literals.append((p.char, resets))
        elif isinstance(p, ConditionalLiteral):
            cond_prefixes.append((p.char, p.var_name, p.condition))
    return joiners, literals, cond_prefixes


def emit_init(emit, instr):
    """Emit __init__ method with hardcoded decode logic."""
    from architecture import ImmediateVariable, TableLookupVariable, ScaledVariable

    emit("    def __init__(self, data: bytes, address: int):")
    emit("        self.address = address")
    emit("        self._resolved_types = {}")
    emit("        val = int.from_bytes(data[:16], 'little')")

    for var_name, var in instr.variables.items():
        safe_name = sanitize_var_name(var_name)
        ranges = tuple(var.bitfield.ranges)
        width = var.bitfield.width
        extract_code = f"self._extract(data, {ranges!r})"

        if isinstance(var, TableLookupVariable):
            table_name = var.table_name
            arg_index = var.arg_index
            emit(f"        self.{safe_name} = TABLES[{table_name!r}].reverse[{extract_code}][{arg_index}]")

        elif isinstance(var, ImmediateVariable):
            emit(f"        _raw = {extract_code}")
            if var.conversion:
                # Emit if/elif chain for conversion
                conv = var.conversion
                first = True
                for comparisons, result_type in conv.cases:
                    # Build OR condition from comparisons
                    conds = []
                    for comp in comparisons:
                        lhs = comp.lhs if isinstance(comp.lhs, int) else f"self.{sanitize_var_name(comp.lhs)}"
                        conds.append(f"{lhs} == {comp.rhs}")
                    cond_str = " or ".join(conds)
                    if first:
                        emit(f"        if {cond_str}:")
                        first = False
                    else:
                        emit(f"        elif {cond_str}:")
                    emit(f"            _imm_type = {result_type!r}")
                emit(f"        else:")
                emit(f"            _imm_type = {conv.default!r}")
            else:
                emit(f"        _imm_type = {var.imm_type!r}")
            # For integer types, use format_width; for floats, use FLOAT_NATIVE_WIDTHS
            if var.imm_type in ('SImm', 'UImm', 'RSImm', 'BITSET'):
                native_width = var.format_width if var.format_width else width
            else:
                native_width = f"FLOAT_NATIVE_WIDTHS.get(_imm_type, {width})"
            emit(f"        _native_width = {native_width}")
            emit(f"        _raw = _raw << (_native_width - {width})")
            emit(f"        if _imm_type in ('SImm', 'RSImm'):")
            emit(f"            if _raw & (1 << (_native_width - 1)):")
            emit(f"                _raw = _raw - (1 << _native_width)")
            emit(f"        if _imm_type == 'RSImm':")
            emit(f"            _raw = address + 16 + _raw")
            emit(f"        self.{safe_name} = _raw")
            emit(f"        self._resolved_types[{var_name!r}] = _imm_type")

        elif isinstance(var, ScaledVariable):
            scale = var.scale
            multiply = var.multiply
            enum_name = var.enum_name
            emit(f"        _raw = {extract_code}")
            if enum_name.startswith('S') or enum_name.startswith('RS'):
                emit(f"        if _raw & (1 << {width - 1}):")
                emit(f"            _raw = _raw - (1 << {width})")
            if scale != 1 or multiply != 1:
                emit(f"        _raw = (_raw * {scale}) // {multiply}")
            if enum_name.startswith('RS'):
                emit(f"        _raw = address + 16 + _raw")
            emit(f"        self.{safe_name} = _raw")

        else:
            emit(f"        self.{safe_name} = {extract_code}")

    emit()


def emit_str(emit, instr, arch):
    """Emit __str__ method using helper function calls."""
    from architecture import Literal, Operand, Opcode, Immediate, Composite, ConditionalLiteral

    elems, _ = arch.parse_format(instr.raw.format)
    mnemonic = instr.mnemonic

    emit("    def __str__(self):")
    emit("        _result = []")
    emit("        _state = self._State()")

    # Handle format_alias
    format_alias = instr.raw.format_alias
    alias_var = None
    if format_alias:
        import re
        alias = format_alias.strip('"').strip("'")
        m = re.match(r'(\w+)\s*=\s*(\w+):(\w+)\(([^)]+)\)', alias)
        if m:
            var_name = m.group(1)
            cast_enum = m.group(2)
            table_name = m.group(3)
            args = [a.strip() for a in m.group(4).split(',')]
            alias_var = var_name
            # Build alias args: use self.var for bitfield vars, constant for implicit vars
            arg_exprs = []
            for a in args:
                if a in instr.variables:
                    arg_exprs.append(f'self.{sanitize_var_name(a)}')
                else:
                    arg_exprs.append(repr(instr.implicit_variables[a]))
            emit(f"        _alias_args = ({', '.join(arg_exprs)},)")
            emit(f"        _{var_name} = None")
            emit(f"        for _keys, _val in TABLES[{table_name!r}].entries:")
            emit(f"            if all(_k is None or _k == _v for _k, _v in zip(_keys, _alias_args)):")
            emit(f"                _{var_name} = ENUMS[{cast_enum!r}].reverse[_val]")
            emit(f"                break")

    decoded_vars = set(instr.variables.keys())
    if alias_var:
        decoded_vars.add(alias_var)

    # Group elements: accumulate prefixes/suffixes for each content element
    pending_prefix = []
    pending_suffix = []

    def get_suffix_lines(elem, arch, decoded_vars, alias_var):
        """Get suffix append code lines."""
        lines = []
        if isinstance(elem, ConditionalLiteral):
            safe_var = sanitize_var_name(f"{elem.var_name}@{elem.condition}")
            lines.append(f"_result.append({elem.char!r} if self.{safe_var} else '')")
        elif isinstance(elem, Literal):
            lines.append(f"_result.append({elem.char!r})")
        elif isinstance(elem, Operand):
            var_name = elem.var_name
            safe_name = sanitize_var_name(var_name) if var_name else None
            enum_name = elem.enum_name
            default = getattr(elem, 'default', None)

            if alias_var and var_name == alias_var:
                if default is not None:
                    lines.append(f"if _{var_name} and _{var_name} != {default!r}:")
                    lines.append(f"    _result.append('.' + _{var_name})")
                else:
                    lines.append(f"if _{var_name}:")
                    lines.append(f"    _result.append('.' + _{var_name})")
                return lines

            if var_name not in decoded_vars:
                if len(arch.enums[enum_name].reverse) == 1:
                    single_numeric = list(arch.enums[enum_name].reverse.keys())[0]
                    single_str = arch.enums[enum_name].reverse[single_numeric]
                    # default is now stored as int, so compare directly
                    if default is None or single_numeric != default:
                        lines.append(f"_result.append('.{single_str}')")
                    return lines

            default_val = elem.default

            if default_val is not None:
                lines.append(f"if self.{safe_name} != {default_val}:")
                lines.append(f"    _result.append('.' + ENUMS[{enum_name!r}].reverse[self.{safe_name}])")
            else:
                lines.append(f"_result.append('.' + ENUMS[{enum_name!r}].reverse[self.{safe_name}])")
        return lines

    def emit_suffix_flush():
        """Emit code to flush pending suffixes if last was visible."""
        if not pending_suffix:
            return
        suffix_lines = []
        for s in pending_suffix:
            lines = get_suffix_lines(s, arch, decoded_vars, alias_var)
            suffix_lines.extend(lines)
        if suffix_lines:
            emit("        if _state.visible:")
            for line in suffix_lines:
                emit("            " + line)

    def has_joiner_prefix(prefixes):
        return any(isinstance(p, Literal) and p.char in [', ', '+'] for p in prefixes)

    for idx, elem in enumerate(elems):
        if elem.role == "prefix":
            emit_suffix_flush()
            pending_suffix = []
            pending_prefix.append(elem)

        elif elem.role == "suffix":
            pending_suffix.append(elem)

        else:
            # Content element
            emit_suffix_flush()
            pending_suffix = []

            joiners, literals, cond_prefixes = get_prefix_info(pending_prefix)

            # Look ahead for suffix ConditionalLiterals that belong to this content
            cond_suffixes = []
            for future_elem in elems[idx + 1:]:
                if future_elem.role == "suffix" and isinstance(future_elem, ConditionalLiteral):
                    cond_suffixes.append((future_elem.char, future_elem.var_name, future_elem.condition))
                elif future_elem.role != "suffix":
                    break  # Stop at next non-suffix

            # Combine prefix and suffix conditions for the modifier check
            all_cond_mods = cond_prefixes + cond_suffixes

            # Build conditional prefix string expression
            if cond_prefixes:
                parts = []
                for char, var_name, condition in cond_prefixes:
                    safe_var = sanitize_var_name(f"{var_name}@{condition}")
                    parts.append(f"({char!r} if self.{safe_var} else '')")
                cond_prefix_expr = ' + '.join(parts)
            else:
                cond_prefix_expr = None

            if isinstance(elem, Opcode):
                emit(f"        # Opcode: {mnemonic}")
                emit(f"        _result.append(' {mnemonic}')")
                emit(f"        _state = self._State(has_prev=False, visible=True, need_space=True, skip_joiner=False)")

            elif isinstance(elem, Literal):
                emit(f"        _result.append({elem.char!r})")
                emit(f"        _state.visible = True")

            elif isinstance(elem, ConditionalLiteral):
                safe_var = sanitize_var_name(f"{elem.var_name}@{elem.condition}")
                emit(f"        _result.append({elem.char!r} if self.{safe_var} else '')")

            elif isinstance(elem, Composite):
                if elem.hidden:
                    emit("        _state.visible = False  # hidden composite")
                else:
                    emit_composite(emit, elem, pending_prefix, arch, decoded_vars)

            elif isinstance(elem, Operand):
                var_name = elem.var_name
                safe_name = sanitize_var_name(var_name) if var_name else None
                enum_name = elem.enum_name
                default_val = elem.default

                # Modifiers are now expanded into ConditionalLiteral elements by parser
                # No need to build mod_prefix/mod_suffix here

                # Check for single-value enum
                if var_name and var_name not in decoded_vars:
                    if len(arch.enums[enum_name].reverse) == 1:
                        single_val = list(arch.enums[enum_name].reverse.values())[0]
                        emit(f"        _content = {single_val!r}")
                        if cond_prefix_expr:
                            emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r}, {cond_prefix_expr})")
                        else:
                            emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r})")
                        pending_prefix = []
                        continue

                emit(f"        # Operand: {var_name}")
                default_str = f"{default_val}" if default_val is not None else "None"
                if all_cond_mods:
                    # If any conditional modifier (prefix or suffix) is active, don't hide the operand
                    cond_check_parts = []
                    for char, cv, condition in all_cond_mods:
                        safe_cv = sanitize_var_name(f"{cv}@{condition}")
                        cond_check_parts.append(f"self.{safe_cv}")
                    cond_check = ' or '.join(cond_check_parts)
                    emit(f"        _has_mod = {cond_check}")
                    emit(f"        _content = self._operand(self.{safe_name}, {enum_name!r}, None if _has_mod else {default_str}, '', '')")
                    if cond_prefix_expr:
                        emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r}, {cond_prefix_expr})")
                    else:
                        emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r})")
                else:
                    emit(f"        _content = self._operand(self.{safe_name}, {enum_name!r}, {default_str}, '', '')")
                    emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r})")
                had_joiner = has_joiner_prefix(pending_prefix)
                if not had_joiner:
                    emit(f"        if not _content: _state.skip_joiner = True  # skip_joiner for hidden operand")

            elif isinstance(elem, Immediate):
                var_name = elem.var_name
                if not var_name:
                    emit("        _state.visible = False")
                    pending_prefix = []
                    continue
                safe_name = sanitize_var_name(var_name)
                default_val = elem.default
                imm_type = elem.imm_type
                width = elem.width

                emit(f"        # Immediate: {var_name}")
                default_str = f"{default_val}" if default_val is not None else "None"
                emit(f"        _type = self._resolved_types.get({var_name!r}, {imm_type!r})")
                emit(f"        _content = self._format_imm(self.{safe_name}, _type, {width}, {default_str})")
                if cond_prefix_expr:
                    emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r}, {cond_prefix_expr})")
                else:
                    emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r})")
                had_joiner = has_joiner_prefix(pending_prefix)
                if not had_joiner:
                    emit(f"        if not _content: _state.skip_joiner = True  # skip_joiner for hidden immediate")

            pending_prefix = []

    # Flush trailing suffixes
    emit_suffix_flush()

    emit("        return ''.join(_result).strip()")
    emit()


def emit_composite(emit, elem, pending_prefix, arch, decoded_vars):
    """Emit code for a visible composite element."""
    from architecture import Literal, Operand, Immediate

    emit("        # Visible composite")
    emit("        _comp_parts = []")
    emit("        _child_visible = False  # last child visible")
    emit("        _comp_has_prev = False  # has previous in composite")

    children = elem.children
    child_prefix = []

    # Process children: group each content element with its trailing suffixes
    i = 0
    while i < len(children):
        child = children[i]
        if child.role == "prefix":
            child_prefix.append(child)
            i += 1
            continue
        elif child.role == "suffix":
            # Suffixes are collected via look-ahead from content elements
            i += 1
            continue

        # Content child - look ahead to collect trailing suffixes
        suffix_copy = []
        j = i + 1
        while j < len(children) and children[j].role == "suffix":
            suffix_copy.append(children[j])
            j += 1

        if isinstance(child, Literal):
            emit(f"        _comp_parts.append({child.char!r})")
            emit("        _child_visible = True")
        elif isinstance(child, Operand):
            var_name = child.var_name
            sanitized_name = sanitize_var_name(var_name) if var_name else None
            enum_name = child.enum_name
            default_val = child.default

            # Check single-value enum
            if var_name and var_name not in decoded_vars:
                if len(arch.enums[enum_name].reverse) == 1:
                    single = list(arch.enums[enum_name].reverse.values())[0]
                    emit(f"        _child_content = {single!r}")
                    emit("        _child_visible = True")
                else:
                    emit(f"        _child_content = ENUMS[{enum_name!r}].reverse[self.{sanitized_name}]")
                    emit("        _child_visible = True")
            else:
                # Build visibility condition: operand OR any suffix non-default
                if default_val is None:
                    # No default for operand - always visible
                    emit("        _child_visible = True")
                else:
                    # Operand has default - check operand OR suffixes for non-default
                    visibility_parts = [f"self.{sanitized_name} != {default_val}"]
                    for suffix in suffix_copy:
                        if isinstance(suffix, Operand):
                            suffix_var = suffix.var_name
                            suffix_sanitized = sanitize_var_name(suffix_var) if suffix_var else None
                            suffix_default = suffix.default
                            if suffix_default is not None:
                                visibility_parts.append(f"self.{suffix_sanitized} != {suffix_default}")
                    emit(f"        _child_visible = {' or '.join(visibility_parts)}")
                emit(f"        _child_content = ENUMS[{enum_name!r}].reverse[self.{sanitized_name}]")

            emit("        if _child_visible:")
            # Emit prefixes
            for prefix in child_prefix:
                if isinstance(prefix, Literal):
                    if prefix.char in [', ', '+']:
                        emit(f"            if _comp_has_prev: _comp_parts.append({prefix.char!r})")
                    else:
                        emit(f"            _comp_parts.append({prefix.char!r})")
                        if prefix.char == '[':
                            emit("            _comp_has_prev = False")
            emit("            _comp_parts.append(_child_content)")
            emit("            _comp_has_prev = True")
            # Emit suffixes for this child
            for suffix in suffix_copy:
                if isinstance(suffix, Literal):
                    emit(f"            _comp_parts.append({suffix.char!r})")
                elif isinstance(suffix, Operand):
                    suffix_var = suffix.var_name
                    suffix_sanitized = sanitize_var_name(suffix_var) if suffix_var else None
                    suffix_enum = suffix.enum_name
                    suffix_default = suffix.default
                    if suffix_default is not None:
                        emit(f"            if self.{suffix_sanitized} != {suffix_default}: _comp_parts.append('.' + ENUMS[{suffix_enum!r}].reverse[self.{suffix_sanitized}])")
                    else:
                        emit(f"            _comp_parts.append('.' + ENUMS[{suffix_enum!r}].reverse[self.{suffix_sanitized}])")

        elif isinstance(child, Immediate):
            cv = child.var_name
            if not cv:
                emit("        _child_visible = False")
                child_prefix = []
                i = j
                continue
            cs = sanitize_var_name(cv)
            ct = child.imm_type
            cw = child.width
            cd = getattr(child, 'default', None)

            if cd is not None:
                emit(f"        _child_visible = self.{cs} != {cd}")
            else:
                emit("        _child_visible = True")
            emit(f"        if _child_visible:")
            emit(f"            _it = self._resolved_types.get({cv!r}, {ct!r})")
            emit(f"            _child_content = self._format_imm(self.{cs}, _it, {cw})")
            emit("        else:")
            emit("            _child_content = ''")
            emit("        if _child_visible:")
            for p in child_prefix:
                if isinstance(p, Literal):
                    if p.char in [', ', '+']:
                        emit(f"            if _comp_has_prev: _comp_parts.append({p.char!r})")
                    else:
                        emit(f"            _comp_parts.append({p.char!r})")
            emit("            _comp_parts.append(_child_content)")
            emit("            _comp_has_prev = True")
            for s in suffix_copy:
                if isinstance(s, Literal):
                    emit(f"            _comp_parts.append({s.char!r})")

        child_prefix = []
        i = j  # Skip past the suffixes we've collected

    # Force first content if empty
    first_content = None
    for child in children:
        if child.role == "" and isinstance(child, (Operand, Immediate)):
            first_content = child
            break

    if first_content:
        emit("        if not _comp_parts:")
        if isinstance(first_content, Operand):
            cv = first_content.var_name
            cs = sanitize_var_name(cv) if cv else None
            ce = first_content.enum_name
            if cv and cv not in decoded_vars and len(arch.enums[ce].reverse) == 1:
                single = list(arch.enums[ce].reverse.values())[0]
                emit(f"            _comp_parts.append({single!r})")
            else:
                emit(f"            _comp_parts.append(ENUMS[{ce!r}].reverse[self.{cs}])")
        elif isinstance(first_content, Immediate):
            cv = first_content.var_name
            cs = sanitize_var_name(cv) if cv else None
            ct = first_content.imm_type
            cw = first_content.width
            emit(f"            _it = self._resolved_types.get({cv!r}, {ct!r})")
            emit(f"            _comp_parts.append(self._format_imm(self.{cs}, _it, {cw}))")

    emit("        _content = ''.join(_comp_parts)")

    # Now emit with prefixes using _emit
    joiners, literals, cond_prefixes = get_prefix_info(pending_prefix)
    if cond_prefixes:
        parts = []
        for char, var_name, condition in cond_prefixes:
            safe_var = sanitize_var_name(f"{var_name}@{condition}")
            parts.append(f"({char!r} if self.{safe_var} else '')")
        cond_prefix_expr = ' + '.join(parts)
        emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r}, {cond_prefix_expr})")
    else:
        emit(f"        self._emit(_result, _content, True, _state, {joiners!r}, {literals!r})")


def emit_operands(emit, instr):
    """Emit operands property."""
    var_names = list(instr.variables.keys())
    if var_names:
        items = ", ".join(f"{n!r}: self.{sanitize_var_name(n)}" for n in var_names)
        emit(f"    @property")
        emit(f"    def operands(self):")
        emit(f"        return {{{items}}}")
    else:
        emit(f"    @property")
        emit(f"    def operands(self):")
        emit(f"        return {{}}")


def generate_disassembler(arch: Architecture, out=sys.stdout):
    """Generate standalone disassembler code."""

    def emit(s=""):
        print(s, file=out)

    # =========================================================================
    # Header
    # =========================================================================
    emit('"""')
    emit("Auto-generated GPU disassembler.")
    emit("")
    emit("Usage:")
    emit("    instr = Instruction(data, address)")
    emit("    print(instr)              # assembly string")
    emit("    print(instr.mnemonic)     # e.g. 'IMAD'")
    emit("    print(instr.operands)     # decoded operand values")
    emit('"""')
    emit()
    emit("import struct")
    emit("import math")
    emit("from dataclasses import dataclass")
    emit()

    # =========================================================================
    # Embedded data
    # =========================================================================
    emit("# " + "=" * 77)
    emit("# Embedded Data")
    emit("# " + "=" * 77)
    emit()

    emit(f"FLOAT_NATIVE_WIDTHS = {repr(FLOAT_NATIVE_WIDTHS)}")
    emit()

    emit('''class _Enum:
    def __init__(self, cases: dict, reverse: dict):
        self.cases = cases
        self.reverse = reverse

class _Table:
    def __init__(self, entries: list, reverse: dict):
        self.entries = entries
        self.reverse = reverse
''')


    # ENUMS
    emit("ENUMS = {")
    for name, enum in sorted(arch.enums.items()):
        cases_str = repr(dict(enum.cases))
        reverse_str = repr(dict(enum.reverse))
        emit(f"    {name!r}: _Enum({cases_str}, {reverse_str}),")
    emit("}")
    emit()

    # TABLES
    emit("TABLES = {")
    for name, table in sorted(arch.tables.items()):
        entries_str = repr(list(table.entries))
        reverse_str = repr(dict(table.reverse))
        emit(f"    {name!r}: _Table({entries_str}, {reverse_str}),")
    emit("}")
    emit()

    # =========================================================================
    # Utility functions
    # =========================================================================
    emit("# " + "=" * 77)
    # =========================================================================
    # Instruction classes
    # =========================================================================
    emit("# " + "=" * 77)
    emit("# Instruction Classes")
    emit("# " + "=" * 77)
    emit()

    def sort_key(instr):
        is_pseudo = '_pseudo' in instr.raw.cls
        num_constraints = sum(1 for bf in instr.bitfields if isinstance(bf.constraint, int) and bf.constraint != instr.opcode)
        return (not is_pseudo, -num_constraints, instr.raw.cls)

    opcode_ranges = {}
    for instr in arch.instructions:
        opcode_value = instr.opcode
        if opcode_value not in opcode_ranges:
            for bf in instr.bitfields:
                if bf.constraint == opcode_value:
                    opcode_ranges[opcode_value] = tuple(bf.ranges)
                    break

    emit('''class Instruction:
    """Base class for decoded instructions."""
    mnemonic: str = ""
    size: int = 16

    @dataclass
    class _State:
        """Formatting state for render_with_roles logic."""
        has_prev: bool = False
        visible: bool = False
        need_space: bool = False
        skip_joiner: bool = False

    @staticmethod
    def _extract(data: bytes, ranges: tuple) -> int:
        """Extract bits from 128-bit instruction data."""
        value = int.from_bytes(data[:16], 'little')
        result = 0
        bit_pos = 0
        for high, low in reversed(ranges):
            width = high - low + 1
            mask = (1 << width) - 1
            bits = (value >> low) & mask
            result |= bits << bit_pos
            bit_pos += width
        return result

    def __new__(cls, data: bytes, address: int = 0):
        if cls is Instruction:
            for opcode_val, parent_cls in OPCODES.items():
                ranges = parent_cls._OPCODE_RANGES
                if Instruction._extract(data, ranges) == opcode_val:
                    return parent_cls(data, address)
            val = int.from_bytes(data[:16], 'little')
            raise ValueError(f"Unknown instruction: 0x{val:032x}")
        return super().__new__(cls)

    @property
    def operands(self) -> dict:
        return {}

    def _operand(self, val, enum_name, default_val, prefix, suffix):
        """Render operand content. Returns string or '' if default."""
        if val == default_val and not prefix and not suffix:
            return ''
        return prefix + ENUMS[enum_name].reverse[val] + suffix

    @staticmethod
    def _format_float(val: float, bits: int, sign_bit: int) -> str:
        if math.isnan(val):
            # nvdisasm only distinguishes SNAN/QNAN for F64, always shows QNAN for F16/F32
            if sign_bit == 0x8000000000000000:  # F64
                quiet_bit = 0x0008000000000000
                nan_type = "QNAN" if bits & quiet_bit else "SNAN"
            else:
                nan_type = "QNAN"
            return f"-{nan_type} " if bits >= sign_bit else f"+{nan_type} "
        if math.isinf(val):
            return "+INF " if val > 0 else "-INF "
        if bits == sign_bit:
            return "-0.0 "
        if bits == 0:
            return "0"
        if abs(val) >= 1e9:
            return f"{val:.20e}"
        if val == int(val):
            return str(int(val))
        return f"{val:.20g}"

    @staticmethod
    def _f16_to_float(bits: int) -> float:
        sign = (bits >> 15) & 1
        exp = (bits >> 10) & 0x1F
        mant = bits & 0x3FF
        if exp == 0x1F:
            return float('-inf') if sign and mant == 0 else (float('inf') if mant == 0 else float('nan'))
        elif exp == 0:
            if mant == 0:
                return -0.0 if sign else 0.0
            val = (mant / 1024) * (2 ** -14)
            return -val if sign else val
        val = (1 + mant / 1024) * (2 ** (exp - 15))
        return -val if sign else val

    @staticmethod
    def _e6m9_to_float(bits: int) -> float:
        sign = (bits >> 15) & 1
        exp = (bits >> 9) & 0x3F
        mant = bits & 0x1FF
        if exp == 0x3F:
            if mant == 0:
                return float('-inf') if sign else float('inf')
            return float('nan')
        elif exp == 0:
            if mant == 0:
                return -0.0 if sign else 0.0
            val = (mant / 512) * (2 ** -30)
            return -val if sign else val
        val = (1 + mant / 512) * (2 ** (exp - 31))
        return -val if sign else val

    def _format_imm(self, val, imm_type, width, default=None):
        """Format immediate value based on type. Returns '' if val equals default."""
        if val == default:
            return ''
        if imm_type == 'BITSET':
            if val == 0:
                return ""
            set_bits = [str(i) for i in range(width - 1, -1, -1) if val & (1 << i)]
            return "{" + ",".join(set_bits) + "}"
        elif imm_type == 'SImm':
            mask = (1 << width) - 1
            bits = val & mask
            sign_bit = 1 << (width - 1)
            if bits >= sign_bit:
                magnitude = ((~bits) + 1) & mask
                return f"-0x{magnitude:x}"
            return f"0x{bits:x}"
        elif imm_type == 'UImm':
            mask = (1 << width) - 1
            return f"0x{val & mask:x}"
        elif imm_type == 'RSImm':
            if val < 0:
                return f"-0x{-val:x}"
            return f"0x{val:x}"
        elif imm_type == 'F16Imm':
            bits = val & 0xFFFF
            fval = self._f16_to_float(bits)
            return self._format_float(fval, bits, 0x8000)
        elif imm_type == 'E8M7Imm':
            bits = val & 0xFFFF
            f32_bits = bits << 16
            fval = struct.unpack('<f', struct.pack('<I', f32_bits))[0]
            return self._format_float(fval, bits, 0x8000)
        elif imm_type == 'E6M9Imm':
            bits = val & 0xFFFF
            fval = self._e6m9_to_float(bits)
            return self._format_float(fval, bits, 0x8000)
        elif imm_type == 'F32Imm':
            bits = val & 0xFFFFFFFF
            fval = struct.unpack('<f', struct.pack('<I', bits))[0]
            return self._format_float(fval, bits, 0x80000000)
        elif imm_type == 'F64Imm':
            bits = val & 0xFFFFFFFFFFFFFFFF
            fval = struct.unpack('<d', struct.pack('<Q', bits))[0]
            return self._format_float(fval, bits, 0x8000000000000000)
        else:
            return f"0x{val:x}"

    def _emit(self, result, content, is_operand, state, joiners, literals, cond_prefix=''):
        """Emit content element with prefix handling. Mutates state."""
        if not content:
            state.visible = False
            return

        if state.need_space and is_operand:
            result.append(' ')
            state.need_space = False

        had_joiner = False
        for j in joiners:
            if state.has_prev and not state.skip_joiner:
                result.append(j)
                had_joiner = True
            state.skip_joiner = False

        for char, resets in literals:
            result.append(char)
            if resets:
                state.has_prev = False

        if is_operand and state.has_prev and not had_joiner:
            result.append(' ')

        # Emit conditional prefixes (-, !, ~, |) before content
        if cond_prefix:
            result.append(cond_prefix)

        result.append(content)

        if is_operand:
            state.has_prev = True
            state.skip_joiner = False

        state.visible = True
''')

    opcode_to_parent = {}

    for mnemonic, instrs in sorted(arch.by_mnemonic.items()):
        instrs = sorted(instrs, key=sort_key)
        assert_instructions_distinguishable(instrs)
        unique_opcodes = sorted(set(i.opcode for i in instrs))
        first_opcode = unique_opcodes[0]
        ranges = opcode_ranges.get(first_opcode, ((23, 12),))
        parent_class_name = mnemonic.replace('.', '_')

        for opcode_val in unique_opcodes:
            opcode_to_parent[opcode_val] = (mnemonic, parent_class_name)

        def get_constraints(instr):
            constraints = []
            instr_opcode = instr.opcode
            instr_ranges = opcode_ranges.get(instr_opcode, ranges)
            if len(unique_opcodes) > 1:
                constraints.append((instr_ranges, 'int', instr_opcode))
            for bf in instr.bitfields:
                if tuple(bf.ranges) == instr_ranges and bf.constraint == instr_opcode:
                    continue
                c = bf.constraint
                if isinstance(c, int):
                    constraints.append((tuple(bf.ranges), 'int', c))
                elif isinstance(c, range):
                    constraints.append((tuple(bf.ranges), 'range', (c.start, c.stop)))
                elif isinstance(c, set):
                    constraints.append((tuple(bf.ranges), 'set', c))
            return constraints

        if len(instrs) == 1:
            instr = instrs[0]
            emit(f"class {parent_class_name}(Instruction):")
            emit(f"    mnemonic = {mnemonic!r}")
            emit(f"    _OPCODE_RANGES = {ranges!r}")
            emit()
            emit_init(emit, instr)
            emit_str(emit, instr, arch)
            emit_operands(emit, instr)
            emit()
        else:
            emit(f"class {parent_class_name}(Instruction):")
            emit(f"    mnemonic = {mnemonic!r}")
            emit(f"    _OPCODE_RANGES = {ranges!r}")
            emit()
            emit(f"    def __new__(cls, data: bytes, address: int):")
            emit(f"        if cls is {parent_class_name}:")
            emit(f"            val = int.from_bytes(data[:16], 'little')")

            for instr in instrs:
                leaf_name = make_class_name(instr.raw.cls)
                constraints = get_constraints(instr)

                if constraints:
                    conditions = []
                    for c_ranges, c_type, c_value in constraints:
                        extract = f"cls._extract(data, {c_ranges!r})"
                        if c_type == 'int':
                            cond = f"{extract} == {c_value}"
                        elif c_type == 'range':
                            start, stop = c_value
                            cond = f"{start} <= {extract} < {stop}"
                        else:
                            cond = f"{extract} in {c_value!r}"
                        conditions.append(cond)
                    condition_str = " and ".join(conditions)
                    emit(f"            if {condition_str}:")
                    emit(f"                return {leaf_name}(data, address)")
                else:
                    emit(f"            return {leaf_name}(data, address)")
                    break

            emit(f"        return object.__new__(cls)")
            emit()

            for instr in instrs:
                leaf_name = make_class_name(instr.raw.cls)
                emit(f"class {leaf_name}({parent_class_name}):")
                emit_init(emit, instr)
                emit_str(emit, instr, arch)
                emit_operands(emit, instr)
                emit()

    # OPCODES table
    emit("# " + "=" * 77)
    emit("# Opcode Dispatch Table")
    emit("# " + "=" * 77)
    emit()
    emit("OPCODES = {")
    for opcode_val, (mnemonic, class_name) in sorted(opcode_to_parent.items()):
        emit(f"    {opcode_val}: {class_name},  # {mnemonic}")
    emit("}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate standalone disassembler")
    parser.add_argument("spec_file", help="Input spec file")
    parser.add_argument("output_file", nargs="?", help="Output Python file")
    args = parser.parse_args()

    with open(args.spec_file, 'r') as f:
        content = f.read()

    arch = Architecture(content)

    if args.output_file:
        with open(args.output_file, 'w') as out:
            generate_disassembler(arch, out)
    else:
        generate_disassembler(arch)
