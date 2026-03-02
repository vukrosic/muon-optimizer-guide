---
name: translate-to-chinese-pdf
description: Translates a LaTeX file to Chinese and generates a PDF. Use when user says chinese, or translate to chinese or ch.
---

# Translate to Chinese PDF Skill

Translates an existing LaTeX file to Chinese and compiles it into a high-quality PDF.

## When to use this skill

- When the user wants a Chinese version of a LaTeX document.
- When the user asks to "translate to chinese" or "chinese pdf".

## Workflow

1. Translate the LaTeX content to Chinese.
   - Maintain the original tone, style, and mathematical rigor.
   - Preserve LaTeX commands, environments, and math formulas exactly.
   - Try to be as accurate to the original file as possible.
2. Setup Chinese LaTeX & Style:
   - Use `\documentclass{ctexart}` or add `\usepackage[UTF8]{ctex}`.
   - Use `xelatex` for compilation.
   - **Styling Rules**:
     - Use `geometry` to set margin to `1.2in`.
     - Use `setspace` to set `\setstretch{1.15}`.
     - Add `\usepackage{parskip}` for paragraph spacing.
     - Disable section numbering using `\setcounter{secnumdepth}{0}`.
3. Compile:
   - Save the translated file with a `-ch.tex` suffix.
   - Run `xelatex` twice to resolve references.

## Commands

```bash
# Compile command (run twice)
xelatex -interaction=nonstopmode "filename-ch.tex"
```

## Tone & Style Guidelines

- If some terminelogy is in english but there is no accurate translation in chinese, you can translate it and put original termin in english in brackets.