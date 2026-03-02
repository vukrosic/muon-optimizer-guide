---
name: md-to-pdf
description: Converts a markdown file to a PDF by first creating a LaTeX file and then compiling it. Invoke if user just says "pdf".
---

# Md to PDF Skill

Converts a markdown file to a PDF by first creating an intermediate LaTeX file and then compiling it using `xelatex`.
Convert word for word, everything must be IDENTICAL. Make spaces between paragraphs nice and proper for the document to look good.

## When to use this skill

- When the user asks to convert a markdown file to PDF or simply says "pdf" while a markdown file is active.

## Workflow

1. **MD to TEX**: Convert the markdown content to a structured LaTeX (`.tex`) file word for word.
   - **Title**: Add `\\ \large Research Progress Report` to the main title.
   - **Author**: Set the author to **Vuk Rosić \thanks{Thank you to Novita AI for providing compute for this research.} \\ \small \raisebox{-0.2\height}{\includegraphics[width=0.02\textwidth]{/Users/vukrosic/AI Science Projects/AI Blog Writing/.agent/skills/md-to-pdf/github.jpg}} \texttt{vukrosic/qk\_norm\_collapse}**.
   - Use standard packages: `fontspec` (for Unicode), `amsmath`, `amssymb`, `hyperref`, `geometry`, `booktabs`, `enumitem`, `parskip`, `setspace`, `graphicx`, `xcolor`, `pagecolor`.
   - Images in tex must be at the exact same spot and order as in the markdown file.
   - **Styling Rules**:
     - Set page margin to `1.2in`.
     - Set line spacing to `1.15` using `\setstretch{1.15}`.
     - Disable section numbering using `\setcounter{secnumdepth}{0}`.
     - **Never include a date**: Ensure `\date{}` is used.
     - **Dark Mode Styling**:
       - Define colors: `\definecolor{darkbg}{HTML}{1E1E1E}` and `\definecolor{lighttext}{HTML}{D4D4D4}`.
       - Apply: `\pagecolor{darkbg}` and `\color{lighttext}`.
       - Configure `hyperref`: `[colorlinks=true, linkcolor=cyan, citecolor=cyan, filecolor=cyan, urlcolor=cyan]`.
     - Do not use horizontal lines (`\hrule`).
2. **TEX to PDF**: Compile the `.tex` file using `xelatex`.
   - Run the command twice to ensure all references and PDF metadata are generated.

## Commands

```bash
# Compile command (run twice)
xelatex -interaction=nonstopmode "filename.tex"
```