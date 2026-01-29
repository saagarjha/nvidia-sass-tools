#!/usr/bin/env python3
"""Mutation tester for generated disassembler.

Generates test cases by randomly selecting fields to vary, then tests
against nvdisasm in parallel.
"""

import importlib
import itertools
import random
import subprocess
import tempfile
import os
import re
from multiprocessing import Pool, cpu_count

from architecture import Architecture

# Global state for workers
_instruction_cls = None


def init_worker(module_name):
    """Initialize worker with generated disassembler."""
    global _instruction_cls
    module = importlib.import_module(module_name)
    _instruction_cls = module.Instruction


def count_options(constraint):
    """Count how many distinct values a constraint allows."""
    if constraint is None:
        return 1
    if isinstance(constraint, int):
        return 1
    if isinstance(constraint, range):
        return len(constraint)
    if isinstance(constraint, set):
        return len(constraint)
    return 1


def get_constraint_values(constraint, max_vals: int = 2):
    """Get list of random valid values for a constraint."""
    if constraint is None:
        return [0]
    if isinstance(constraint, int):
        return [constraint]
    if isinstance(constraint, range):
        if len(constraint) <= max_vals:
            return list(constraint)
        # Return random sample from range
        return random.sample(range(constraint.start, constraint.stop), max_vals)
    if isinstance(constraint, set):
        vals = list(constraint)
        if len(vals) <= max_vals:
            return vals
        return random.sample(vals, max_vals)
    return [0]


def get_variable_fields(instr):
    """Get list of (bf_index, num_options, desc) for variable fields."""
    fields = []
    for i, bf in enumerate(instr.bitfields):
        num_options = count_options(bf.constraint)
        if num_options <= 1:
            continue
        desc = bf.variables[0].name if bf.variables else f"bits{bf.ranges}"
        fields.append((i, num_options, desc))
    return fields


def build_instruction_bytes(instr, field_values: dict) -> bytes:
    """Build 16-byte instruction from field values."""
    val = 0
    for i, bf in enumerate(instr.bitfields):
        if i in field_values:
            field_val = field_values[i]
        else:
            values = get_constraint_values(bf.constraint)
            field_val = values[0]

        bit_pos = 0
        for high, low in reversed(bf.ranges):
            width = high - low + 1
            mask = (1 << width) - 1
            bits = (field_val >> bit_pos) & mask
            val |= bits << low
            bit_pos += width

    return val.to_bytes(16, 'little')


def generate_test_cases(arch, seed: int, per_instr_cap: int = 1024, budget: int = None):
    """Generate test cases with per-instruction cap."""
    random.seed(seed)

    # For each instruction, randomly select fields until we hit per_instr_cap
    test_cases = []

    for instr in arch.instructions:
        fields = get_variable_fields(instr)
        if not fields:
            test_cases.append((instr, {}))
            continue

        random.shuffle(fields)

        # Select fields until product exceeds per_instr_cap
        selected = []
        product = 1
        for bf_idx, num_options, desc in fields:
            pick = min(2, num_options)
            if product * pick > per_instr_cap and selected:
                break
            selected.append((bf_idx, num_options))
            product *= pick

        # Generate combinations
        if not selected:
            test_cases.append((instr, {}))
        else:
            value_lists = []
            bf_indices = []
            for bf_idx, num_options in selected:
                bf = instr.bitfields[bf_idx]
                all_vals = get_constraint_values(bf.constraint)
                # Pick first and last (or just first if only one)
                test_vals = [all_vals[0], all_vals[-1]] if len(all_vals) > 1 else [all_vals[0]]
                value_lists.append(test_vals)
                bf_indices.append(bf_idx)

            for combo in itertools.product(*value_lists):
                field_values = dict(zip(bf_indices, combo))
                test_cases.append((instr, field_values))

        # Early exit if we've hit the budget
        if budget and len(test_cases) >= budget:
            break

    return test_cases[:budget] if budget else test_cases


def process_batch(args) -> dict:
    """Process a batch of test cases against nvdisasm."""
    batch_bytes, batch_classes, batch_start = args

    # Write batch to temp file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        for instr_bytes in batch_bytes:
            f.write(instr_bytes)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["nvdisasm", "-b", "SM90", tmp_path],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            os.unlink(tmp_path)
            return {"error": result.stderr, "tested": 0, "differences": []}

        # Parse nvdisasm output: /*offset*/ instruction ;
        pattern = r'/\*([0-9a-f]+)\*/\s+(.+?)\s*;'
        nvdisasm_results = {}
        for m in re.finditer(pattern, result.stdout, re.IGNORECASE):
            offset = int(m.group(1), 16)
            asm = m.group(2).strip()
            nvdisasm_results[offset] = asm

        # Compare with our disassembler
        differences = []
        tested = 0

        for i, instr_bytes in enumerate(batch_bytes):
            offset = i * 16
            expected = nvdisasm_results.get(offset, "NOT_FOUND")

            try:
                instr = _instruction_cls(instr_bytes, offset)
                actual = str(instr)
            except Exception as e:
                actual = f"ERROR: {e}"

            tested += 1
            # If nvdisasm says NOT_FOUND, treat as pass (masked instructions)
            if expected == "NOT_FOUND":
                continue
            if actual != expected:
                differences.append({
                    "offset": batch_start + offset,
                    "expected": expected,
                    "actual": actual,
                    "cls": batch_classes[i]
                })

        os.unlink(tmp_path)
        return {"tested": tested, "differences": differences, "error": None}

    except subprocess.TimeoutExpired:
        os.unlink(tmp_path)
        return {"error": "timeout", "tested": 0, "differences": []}
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return {"error": str(e), "tested": 0, "differences": []}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mutation tester for disassembler")
    parser.add_argument("module", nargs="?", default="sm90_disasm",
                        help="Disassembler module name")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed (default: random)")
    parser.add_argument("-b", "--budget", type=int, default=None,
                        help="Total test budget (default: unlimited)")
    parser.add_argument("-c", "--per-instr-cap", type=int, default=10_000,
                        help="Max combinations per instruction (default: 10K)")
    parser.add_argument("-j", "--jobs", type=int, default=cpu_count(),
                        help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Instructions per nvdisasm batch")
    parser.add_argument("--max-diff", type=int, default=20,
                        help="Max differences to show")
    parser.add_argument("--spec", default="/home/ubuntu/sm_90_instructions.txt",
                        help="Architecture spec file")
    args = parser.parse_args()

    # Generate or use provided seed
    if args.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed

    print(f"Seed: {seed}")
    print(f"Budget: {args.budget:,}" if args.budget else "Budget: unlimited")
    print(f"Per-instruction cap: {args.per_instr_cap:,}")
    print(f"Module: {args.module}")
    print(f"Workers: {args.jobs}")
    print()

    # Load architecture
    print("Loading architecture spec...")
    with open(args.spec, 'r') as f:
        content = f.read()
    arch = Architecture(content)
    print(f"Loaded {len(arch.instructions)} instructions")

    # Generate test cases
    print("Generating test cases...")
    test_cases = generate_test_cases(arch, seed, args.per_instr_cap, args.budget)
    print(f"Generated {len(test_cases):,} test cases")

    # Build instruction bytes
    print("Building instruction bytes...")
    test_bytes = []
    test_classes = []  # Track class names
    for instr, field_values in test_cases:
        test_bytes.append(build_instruction_bytes(instr, field_values))
        test_classes.append(instr.raw.cls)
    print(f"Built {len(test_bytes):,} instruction byte sequences")

    # Split into batches
    batches = []
    for i in range(0, len(test_bytes), args.batch_size):
        batch_bytes = test_bytes[i:i + args.batch_size]
        batch_classes = test_classes[i:i + args.batch_size]
        batches.append((batch_bytes, batch_classes, i * 16))

    print(f"Processing {len(batches)} batches with {args.jobs} workers...")

    # Process in parallel
    with Pool(args.jobs, initializer=init_worker, initargs=(args.module,)) as pool:
        results = pool.map(process_batch, batches)

    # Aggregate results
    total_tested = 0
    total_diff = 0
    total_errors = 0
    diff_counts = {}  # (expected, actual) -> (count, set of classes)

    for r in results:
        if r["error"]:
            total_errors += 1
            continue

        total_tested += r["tested"]
        total_diff += len(r["differences"])

        for d in r["differences"]:
            key = (d["expected"], d["actual"])
            if key not in diff_counts:
                diff_counts[key] = [0, set()]
            diff_counts[key][0] += 1
            diff_counts[key][1].add(d["cls"])

    # Print summary
    print()
    print("=" * 60)
    print(f"Seed: {seed}")
    print(f"Total batches: {len(batches)}")
    print(f"Total errors: {total_errors}")
    print(f"Total tested: {total_tested:,}")
    print(f"Total differences: {total_diff:,}")
    if total_tested > 0:
        accuracy = (total_tested - total_diff) / total_tested * 100
        print(f"Accuracy: {accuracy:.4f}%")

    # Show top differences with classes
    if diff_counts:
        print(f"\nTop difference patterns:")
        sorted_diffs = sorted(diff_counts.items(), key=lambda x: -x[1][0])
        for (expected, actual), (count, classes) in sorted_diffs[:args.max_diff]:
            classes_str = ", ".join(sorted(classes)[:5])
            if len(classes) > 5:
                classes_str += f", ... (+{len(classes)-5} more)"
            print(f"  [{count}x] classes: {classes_str}")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")


if __name__ == "__main__":
    main()
