#!/usr/bin/env python3

import os, time, json, re, subprocess, tempfile, csv, ast
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency, f_oneway
from itertools import combinations
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit only this block
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = "xx"

DATASET_CSV   = "problems_dataset_150.csv"
PHASE1_CSV    = "phase1_llama70b-2.csv"
PHASE2_CSV    = "phase2_qwen3.csv"
CHI_SQUARE_CSV = "chi_square_report.csv"

PHASE1_MODEL  = "llama-3.3-70b-versatile"   # free, 70B, strong
PHASE2_MODEL  = "qwen3-32b"                  # free on Groq, thinking model

MAX_TOKENS    = 2048    # kept low to save tokens — sufficient for a solution function
TEMPERATURE   = 0
SLEEP_P1      = 6       # 6k TPM free limit — safe buffer
SLEEP_P2      = 6

groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(strategy, name, description, examples):
    ex = "\n".join([
        f"Input: {e.get('input', '')}\nOutput: {e.get('output', '')}"
        for e in examples[:2]
    ])
    if strategy == "P1":
        return (
            f"Solve the following coding problem in Python.\n\n"
            f"Problem: {name}\n{description}\n\n"
            f"Examples:\n{ex}\n\n"
            f"Return ONLY a Python function named 'solution'. No explanation."
            f"Write a complete Python program that reads from stdin and prints to stdout. No function definitions. No explanations.\n\n"
        )
    elif strategy == "P2":
        return (
            f"You are an expert Python programmer.\n\n"
            f"Requirements:\n"
            f"  - Handle all edge cases and constraints\n"
            f"  - Optimise for time and space complexity\n"
            f"  - Return ONLY a Python function named 'solution'\n\n"
            f"Problem: {name}\n{description}\n\n"
            f"Examples:\n{ex}"
            f"Write a complete Python program that reads from stdin and prints to stdout. No function definitions. No explanations.\n\n"
        )
    else:  # P3 — Chain of Thought
        return (
            f"Solve this coding problem step by step in Python.\n\n"
            f"Step 1 - Restate the problem in your own words.\n"
            f"Step 2 - Identify all constraints and edge cases.\n"
            f"Step 3 - Choose the optimal algorithm and data structure.\n"
            f"Step 4 - Trace through the example to verify your approach.\n"
            f"Step 5 - Write pseudocode.\n"
            f"Step 6 - Implement the solution in Python (function named 'solution').\n"
            f"Step 7 - Verify correctness against examples and edge cases.\n\n"
            f"Problem: {name}\n{description}\n\n"
            f"Examples:\n{ex}\n\n"
            f"Return ONLY the final Python function from Step 6."
            f"Write a complete Python program that reads from stdin and prints to stdout. No function definitions. No explanations.\n\n"
        )


def build_sanity_prompt(name, description, examples, difficulty):
    ex = "\n".join([
        f"Input: {e.get('input', '')}\nOutput: {e.get('output', '')}"
        for e in examples[:2]
    ])
    return (
        f"This is a {difficulty}-level problem that another model failed to solve.\n"
        f"Solve it carefully in Python.\n\n"
        f"Problem: {name}\n{description}\n\n"
        f"Examples:\n{ex}\n\n"
        f"Return ONLY a Python function named 'solution'."
    )

# ─────────────────────────────────────────────────────────────────────────────
# LIVECODEBENCH-STYLE MULTI-LAYER OUTPUT CHECKER
# ─────────────────────────────────────────────────────────────────────────────
def lcb_check(actual: str, expected: str) -> bool:
    """
    Layer 1 : Exact match after strip
    Layer 2 : Boolean normalisation  True/False → true/false
    Layer 3 : Numeric tolerance      floats within 1e-6
    Layer 4 : AST object equality    lists, tuples, dicts
    Layer 5 : Set / order-free list  equality
    Layer 6 : JSON parse equality
    Layer 7 : Multi-line recursive   compare line by line
    """
    a = actual.strip()
    e = expected.strip()

    # L1 — exact
    if a == e:
        return True

    # L2 — boolean normalisation
    a2 = a.replace("True", "true").replace("False", "false")
    e2 = e.replace("True", "true").replace("False", "false")
    if a2 == e2:
        return True

    # L3 — numeric
    try:
        if abs(float(a) - float(e)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # L4 + L5 — AST parse
    try:
        a_obj = ast.literal_eval(a)
        e_obj = ast.literal_eval(e)
        if a_obj == e_obj:
            return True
        if isinstance(a_obj, list) and isinstance(e_obj, list):
            if sorted(str(x) for x in a_obj) == sorted(str(x) for x in e_obj):
                return True
    except Exception:
        pass

    # L6 — JSON
    try:
        if json.loads(a2) == json.loads(e2):
            return True
    except Exception:
        pass

    # L7 — multi-line recursive
    a_lines = a.splitlines()
    e_lines = e.splitlines()
    if len(a_lines) == len(e_lines) > 1:
        if all(lcb_check(al, el) for al, el in zip(a_lines, e_lines)):
            return True

    return False

# ─────────────────────────────────────────────────────────────────────────────
# API CALLER — single function, model passed as argument
# ─────────────────────────────────────────────────────────────────────────────
def call_groq(prompt_text, model, retries=3):
    start = time.time()
    for attempt in range(retries):
        try:
            resp = groq_client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_text}]
            )
            content = resp.choices[0].message.content or ""
            # Strip <think>...</think> tokens (Qwen3 thinking mode + DeepSeek-R1)
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            return content, round(time.time() - start, 2)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 20 * (attempt + 1)
                print(f"\n    [{model}] Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n    [{model}] ERROR: {err}")
                return None, round(time.time() - start, 2)
    return None, round(time.time() - start, 2)

# ─────────────────────────────────────────────────────────────────────────────
# CODE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_code(text):
    if not text:
        return ""
    m = re.search(r"```python\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def check_compile(code):
    if not code:
        return False, "no code"
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER — uses lcb_check comparator
# ─────────────────────────────────────────────────────────────────────────────
def run_tests(code, test_cases):
    if not code or not test_cases:
        return False, "No code or tests", 0, len(test_cases) if test_cases else 0

    passed, errors, fname = 0, [], None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            fname = f.name

        for tc in test_cases:
            try:
                raw_input  = tc["input"].replace("\\\\n", "\n").replace("\\n", "\n")
                raw_expect = tc["expected"].replace("\\\\n", "\n").replace("\\n", "\n").strip()

                proc = subprocess.run(
                    ["python3", fname],
                    input=raw_input,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                actual = proc.stdout.strip()

                if lcb_check(actual, raw_expect):
                    passed += 1
                else:
                    errors.append(f"want='{raw_expect[:50]}' got='{actual[:50]}'")

            except subprocess.TimeoutExpired:
                errors.append("Timeout")
            except Exception as ex:
                errors.append(str(ex))

    finally:
        if fname:
            try:
                os.remove(fname)
            except Exception:
                pass

    return (passed == len(test_cases)), "; ".join(errors[:1]), passed, len(test_cases)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PROBLEMS — 50 Easy + 50 Medium + 50 Hard = 150 total
# ─────────────────────────────────────────────────────────────────────────────
def load_problems():
    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")
    df     = pd.read_csv(DATASET_CSV)
    easy   = df[df["Difficulty"] == "Easy"].head(50).to_dict("records")
    medium = df[df["Difficulty"] == "Medium"].head(50).to_dict("records")
    hard   = df[df["Difficulty"] == "Hard"].head(50).to_dict("records")
    problems = []
    for diff, subset in [("Easy", easy), ("Medium", medium), ("Hard", hard)]:
        for p in subset:
            problems.append({
                "id":           p["ProblemID"],
                "name":         p["ProblemName"],
                "difficulty":   diff,
                "description":  p["Description"],
                "public_tests": json.loads(p["PublicTests"]),
                "hidden_tests": json.loads(p["HiddenTests"]),
            })
    print(f"Loaded: Easy={len(easy)}, Medium={len(medium)}, Hard={len(hard)} → {len(problems)} total")
    return problems

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — llama-3.3-70b-versatile × P1/P2/P3
# ─────────────────────────────────────────────────────────────────────────────
def run_phase1(problems):
    strategies = ["P1", "P2", "P3"]
    total      = len(problems) * len(strategies)
    done       = set()

    if os.path.exists(PHASE1_CSV):
        with open(PHASE1_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["ProblemID"], row["Prompt"]))
        print(f"Resuming Phase 1 — {len(done)} done, {total - len(done)} remaining")

    fields       = ["ProblemID", "ProblemName", "Difficulty", "Prompt", "Model",
                    "Compiled", "PublicPassed", "HiddenPassed", "GenTime_s", "ErrorMsg"]
    write_header = not os.path.exists(PHASE1_CSV) or len(done) == 0

    with open(PHASE1_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        run_n = 0
        for prob in problems:
            for strat in strategies:
                run_n += 1
                if (prob["id"], strat) in done:
                    continue

                print(f"  [{run_n:>3}/{total}] {prob['difficulty']:<7} "
                      f"{prob['name'][:35]:<35} {strat} ", end="", flush=True)

                prompt        = build_prompt(strat, prob["name"],
                                             prob["description"], prob["public_tests"])
                raw, gen_time = call_groq(prompt, PHASE1_MODEL)
                code          = extract_code(raw)
                compiled, ce  = check_compile(code)

                if not compiled or not prob["public_tests"]:
                    pub_ok, pe = False, ce
                else:
                    pub_ok, pe, _, _ = run_tests(code, prob["public_tests"])

                if not pub_ok or not prob["hidden_tests"]:
                    hid_ok, he = False, "not reached"
                else:
                    hid_ok, he, _, _ = run_tests(code, prob["hidden_tests"])

                print(f"C={int(compiled)} P={int(pub_ok)} H={int(hid_ok)} ({gen_time}s)")
                writer.writerow({
                    "ProblemID":    prob["id"],
                    "ProblemName":  prob["name"],
                    "Difficulty":   prob["difficulty"],
                    "Prompt":       strat,
                    "Model":        PHASE1_MODEL,
                    "Compiled":     int(compiled),
                    "PublicPassed": int(pub_ok),
                    "HiddenPassed": int(hid_ok),
                    "GenTime_s":    gen_time,
                    "ErrorMsg":     (ce or pe or he)[:200],
                })
                f.flush()
                time.sleep(SLEEP_P1)

    print(f"\n✅ Phase 1 complete → {PHASE1_CSV}")
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — qwen3-32b on ALL confirmed failures
# Reads from confirmed_failures.csv (after manual FN check)
# ─────────────────────────────────────────────────────────────────────────────
CONFIRMED_FAILURES_CSV = "confirmed_failures.csv"

def run_phase2(problems_map):
    # Gate: confirmed_failures.csv must exist (produced after manual FN check)
    # If it doesn't exist yet, fall back to all Phase 1 failures
    if os.path.exists(CONFIRMED_FAILURES_CSV):
        source_df  = pd.read_csv(CONFIRMED_FAILURES_CSV)
        source_tag = "confirmed_failures.csv"
    elif os.path.exists(PHASE1_CSV):
        source_df  = pd.read_csv(PHASE1_CSV)
        source_df  = source_df[source_df["HiddenPassed"] == 0]
        source_tag = "phase1_llama70b.csv (all failures — FN check not done yet)"
    else:
        print("No input found — run Phase 1 first.")
        return

    total = len(source_df)
    print(f"\nPhase 2 source : {source_tag}")
    print(f"Total to rerun : {total}")
    print(f"Breakdown      :")
    for diff in ["Easy", "Medium", "Hard"]:
        sub = source_df[source_df["Difficulty"] == diff]
        print(f"  {diff}: {len(sub)}")

    # Resume support
    done = set()
    if os.path.exists(PHASE2_CSV):
        with open(PHASE2_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["ProblemID"], row["OriginalPrompt"]))
        print(f"Resuming — {len(done)} done, {total - len(done)} remaining")

    fields = [
        "ProblemID", "ProblemName", "Difficulty",
        "OriginalPrompt", "OriginalModel", "Phase2Model",
        "Compiled", "PublicPassed", "HiddenPassed",
        "GenTime_s", "ErrorMsg"
    ]
    write_header = not os.path.exists(PHASE2_CSV) or len(done) == 0

    with open(PHASE2_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for idx, row in source_df.iterrows():
            pid   = str(row["ProblemID"])
            strat = row["Prompt"]
            prob  = problems_map.get(pid) or problems_map.get(row["ProblemID"])

            if (pid, strat) in done or not prob:
                continue

            print(f"  [{idx+1:>3}/{total}] {row['Difficulty']:<7} "
                  f"{str(row['ProblemName'])[:35]:<35} "
                  f"(was {strat}) Qwen3-32B ... ", end="", flush=True)

            prompt        = build_sanity_prompt(
                prob["name"], prob["description"],
                prob["public_tests"], row["Difficulty"]
            )
            raw, gen_time = call_groq(prompt, PHASE2_MODEL)
            code          = extract_code(raw)
            compiled, ce  = check_compile(code)

            pub_ok, pe = False, ce
            if compiled and prob.get("public_tests"):
                pub_ok, pe, _, _ = run_tests(code, prob["public_tests"])

            hid_ok, he = False, "not reached"
            if pub_ok and prob.get("hidden_tests"):
                hid_ok, he, _, _ = run_tests(code, prob["hidden_tests"])

            print(f"C={int(compiled)} P={int(pub_ok)} H={int(hid_ok)} ({gen_time}s)")
            writer.writerow({
                "ProblemID":      pid,
                "ProblemName":    prob.get("name", ""),
                "Difficulty":     row["Difficulty"],
                "OriginalPrompt": strat,
                "OriginalModel":  row.get("Model", PHASE1_MODEL),
                "Phase2Model":    PHASE2_MODEL,
                "Compiled":       int(compiled),
                "PublicPassed":   int(pub_ok),
                "HiddenPassed":   int(hid_ok),
                "GenTime_s":      gen_time,
                "ErrorMsg":       (ce or pe or he)[:200],
            })
            f.flush()
            time.sleep(SLEEP_P2)

    print(f"\n Phase 2 complete → {PHASE2_CSV}")
# ─────────────────────────────────────────────────────────────────────────────
# STATS — CHI-SQUARE + PAIRWISE FISHER EXACT (no ANOVA)
# ─────────────────────────────────────────────────────────────────────────────
def run_statistics():
    if not os.path.exists(PHASE1_CSV):
        print("Phase 1 CSV not found.")
        return

    df      = pd.read_csv(PHASE1_CSV)
    metrics = ["Compiled", "PublicPassed", "HiddenPassed"]
    diffs   = ["Easy", "Medium", "Hard"]
    prompts = ["P1", "P2", "P3"]
    results = []

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS — CHI-SQUARE + PAIRWISE FISHER EXACT")
    print("="*70)

    for diff in diffs:
        sub = df[df["Difficulty"] == diff]
        n   = len(sub[sub["Prompt"] == "P1"])

        print(f"\n{'─'*70}")
        print(f"DIFFICULTY: {diff}  (n={n} per prompt)")
        print(f"{'─'*70}")

        for metric in metrics:
            groups = [sub[sub["Prompt"] == p][metric].values for p in prompts]
            means  = [g.mean() * 100 for g in groups]

            print(f"\n  {metric}:")
            print(f"    P1={means[0]:.1f}%  P2={means[1]:.1f}%  P3={means[2]:.1f}%")

            # ── Chi-Square (3-way contingency table) ──────────────────────────
            # Rows = prompts (P1, P2, P3), Cols = [pass, fail]
            ct_3way = np.array([
                [int(g.sum()), int(len(g) - g.sum())] for g in groups
            ])
            chi2, p_chi2, dof, expected = chi2_contingency(ct_3way)
            print(f"    Chi²    χ²={chi2:.3f}  df={dof}  p={p_chi2:.4f}  "
                  f"{'✅ Significant' if p_chi2 < 0.05 else '❌ Not significant'}")

            results.append({
                "Difficulty":    diff,
                "Metric":        metric,
                "Test":          "ChiSquare",
                "Comparison":    "P1 vs P2 vs P3",
                "Statistic":     round(chi2, 3),
                "df":            dof,
                "p_value":       round(p_chi2, 4),
                "Significant":   p_chi2 < 0.05,
                "P1_%":          round(means[0], 1),
                "P2_%":          round(means[1], 1),
                "P3_%":          round(means[2], 1),
            })

            # ── Pairwise Fisher Exact (post-hoc) ──────────────────────────────
            for (i, pa), (j, pb) in combinations(enumerate(prompts), 2):
                ga, gb = groups[i], groups[j]
                ct     = np.array([
                    [int(ga.sum()), int(len(ga) - ga.sum())],
                    [int(gb.sum()), int(len(gb) - gb.sum())]
                ])
                odds, p_fish = fisher_exact(ct, alternative="two-sided")
                direction    = ">" if means[i] > means[j] else "<"
                sig_sym      = "✅" if p_fish < 0.05 else "  "
                print(f"    {sig_sym} Fisher {pa} {direction} {pb}:  "
                      f"p={p_fish:.4f}  OR={odds:.2f}")

                results.append({
                    "Difficulty":  diff,
                    "Metric":      metric,
                    "Test":        f"Fisher_{pa}_vs_{pb}",
                    "Comparison":  f"{pa} vs {pb}",
                    "Statistic":   round(odds, 3),
                    "df":          1,
                    "p_value":     round(p_fish, 4),
                    "Significant": p_fish < 0.05,
                    "P1_%":        round(means[0], 1),
                    "P2_%":        round(means[1], 1),
                    "P3_%":        round(means[2], 1),
                })

    # ── Phase 2 Qwen3 summary ─────────────────────────────────────────────────
    if os.path.exists(PHASE2_CSV):
        p2 = pd.read_csv(PHASE2_CSV)
        print("\n" + "="*70)
        print("PHASE 2 — QWEN3-32B RESULTS ON CONFIRMED FAILURES")
        print("="*70)
        for diff in diffs:
            sub2 = p2[p2["Difficulty"] == diff]
            if len(sub2) == 0:
                continue
            h = int(sub2["HiddenPassed"].sum())
            t = len(sub2)
            print(f"  {diff}: Qwen3-32B solved {h}/{t} ({h/t*100:.1f}%)")

        # Cross-prompt breakdown
        print("\n  Breakdown by Original Failing Prompt:")
        for strat in ["P1", "P2", "P3"]:
            sub_s = p2[p2["OriginalPrompt"] == strat]
            if len(sub_s) == 0:
                continue
            h = int(sub_s["HiddenPassed"].sum())
            t = len(sub_s)
            print(f"    Originally failed {strat}: "
                  f"Qwen3 fixed {h}/{t} ({h/t*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(CHI_SQUARE_CSV, index=False)
    print(f"\n✅ Stats saved → {CHI_SQUARE_CSV}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY — Winning Prompt per Difficulty × Metric")
    print("="*70)
    for diff in diffs:
        sub     = df[df["Difficulty"] == diff]
        row_out = f"  {diff:<8}"
        for metric in metrics:
            means  = {p: sub[sub["Prompt"] == p][metric].mean() * 100
                      for p in prompts}
            winner = max(means, key=means.get)
            row_out += f"  {metric}: {winner}({means[winner]:.0f}%)"
        print(row_out)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — this is what actually triggers the pipeline
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Thesis Pipeline — Groq Only")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "stats", "all"],
        default="1",                        # ← defaults to Phase 1 only for safety
        help="1=Phase1  2=Phase2  stats=Chi-Square analysis  all=everything"
    )
    args = parser.parse_args()

    problems     = load_problems()
    problems_map = {p["id"]: p for p in problems}

    if args.phase in ("1", "all"):
        print("\n" + "="*70)
        print(f"PHASE 1 — {PHASE1_MODEL} × P1/P2/P3 × 150 Problems")
        print("="*70)
        run_phase1(problems)

    if args.phase in ("2", "all"):
        print("\n" + "="*70)
        print(f"PHASE 2 — {PHASE2_MODEL} on Confirmed Failures")
        print("="*70)
        run_phase2(problems_map)

    if args.phase in ("stats", "all"):
        run_statistics()

    print("\n🎉 Done!")
    print(f"   {PHASE1_CSV:<30} — Phase 1 results")
    print(f"   {PHASE2_CSV:<30} — Phase 2 results")
    print(f"   {CHI_SQUARE_CSV:<30} — Chi-Square + Fisher stats")