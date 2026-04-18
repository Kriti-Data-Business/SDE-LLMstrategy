
```markdown
# SDE-LLMstrategy 

> A systematic benchmarking framework for evaluating LLM code generation 
> quality across prompting strategies — validated using LiveCodeBench's 
> standard I/O evaluation protocol.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Type-Research%20Pipeline-green.svg)]()
[![RMIT](https://img.shields.io/badge/Institution-RMIT%20University-red.svg)]()

---

##  Overview

As enterprises integrate Large Language Models (LLMs) into software 
development workflows, selecting the optimal prompting strategy directly 
impacts code reliability and developer productivity.

**SDE-LLMstrategy** provides a reproducible, automated evaluation pipeline 
that systematically compares how different prompting strategies affect the 
correctness of LLM-generated code solutions — enabling teams to make 
evidence-based decisions before deploying LLM-assisted coding tools at scale.

---

##  Research Questions

1. Which prompting strategy — **Zero-Shot**, **Few-Shot**, or 
   **Chain-of-Thought (CoT)** — produces the highest pass rate on 
   competitive programming problems?
2. Does prompting strategy effectiveness vary across problem 
   difficulty levels (Easy / Medium / Hard)?
3. What is the performance gap between public test pass rates 
   and hidden test pass rates across strategies?

---

##  Pipeline Architecture

```
problems_dataset_150.csv
        │
        ▼
┌─────────────────────┐
│   Prompt Builder    │  ← Zero-Shot / Few-Shot / Chain-of-Thought
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    LLM Generator    │  ← GPT-4o / Claude / Gemini (configurable)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Test Executor     │  ← stdin/stdout via subprocess (LiveCodeBench protocol)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Results Analyser   │  ← phase1_results.csv / retest_results.csv
└─────────────────────┘
```

---

##  Evaluation Protocol

This project strictly follows **LiveCodeBench's evaluation methodology** 
(Jain et al., 2024) for Codeforces-sourced problems:

- Each LLM-generated solution is executed as a standalone Python program
- Test case inputs are supplied via **stdin**
- Captured **stdout** is compared against expected output
- Both **public tests** (P) and **hidden tests** (H) are evaluated

### Metrics Per Problem

| Flag | Meaning |
|------|---------|
| `C=1` | Code compiled without errors |
| `P=1` | All public test cases passed |
| `H=1` | All hidden test cases passed |

---

##  Dataset

- **150 problems** sourced from Codeforces via LiveCodeBench
- Difficulty distribution: Easy / Medium / Hard
- Each problem includes: problem statement, public test cases, hidden test cases

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Kriti-Data-Business/SDE-LLMstrategy.git
cd SDE-LLMstrategy
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
export OPENAI_API_KEY=your_key_here
```

### 4. Run Phase 1 evaluation

```bash
python pipeline.py --phase 1
```

### 5. View results

```bash
cat phase1_results.csv
```

---

##  Project Structure

```
SDE-LLMstrategy/
│
├── pipeline.py                  # Main evaluation pipeline
├── problems_dataset_150.csv     # Benchmark dataset (150 problems)
├── phase1_results.csv           # Phase 1 output results
├── retest_results.csv           # Retest output results
├── failed_tests.csv             # Failed cases for analysis
├── requirements.txt             
└── README.md
```

---

## Industry Use Cases

This framework is directly applicable in enterprise settings where teams need to:

- **Evaluate AI coding assistants** (GitHub Copilot, Cursor, Amazon CodeWhisperer)
  before adoption
- **Select optimal prompting templates** for internal LLM-based developer tools
- **Benchmark model upgrades** — measure whether a new LLM version improves 
  code correctness
- **Establish quality baselines** for AI-generated code in CI/CD pipelines
- **Audit prompt strategy ROI** across different engineering problem domains

---

##  Results Summary

*(To be updated after full pipeline run)*

| Strategy | Easy P% | Easy H% | Medium P% | Medium H% | Hard P% | Hard H% |
|----------|---------|---------|-----------|-----------|---------|---------|
| Zero-Shot | - | - | - | - | - | - |
| Few-Shot | - | - | - | - | - | - |
| Chain-of-Thought | - | - | - | - | - | - |

---

##  Citation

If you use this framework in your research, please cite:

```bibtex
@misc{yadav2026sde,
  title     = {SDE-LLMstrategy: Benchmarking LLM Code Generation 
               Across Prompting Strategies},
  author    = {Yadav, Kriti},
  year      = {2026},
  institution = {RMIT University},
  url       = {https://github.com/Kriti-Data-Business/SDE-LLMstrategy}
}
```

**LiveCodeBench reference:**
```bibtex
@article{jain2024livecodebench,
  title  = {LiveCodeBench: Holistic and Contamination Free 
            Evaluation of Large Language Models for Code},
  author = {Jain et al.},
  year   = {2024}
}
```

---

## 🔬 Research Context

This project is developed as part of a **Master of Data Science** 
thesis at **RMIT University**, investigating the impact of prompt 
engineering strategies on LLM code generation reliability for 
competitive programming tasks.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contact

**Kriti Yadav**  
Master of Data Science — RMIT University, Melbourne  
[GitHub](https://github.com/Kriti-Data-Business)
```
