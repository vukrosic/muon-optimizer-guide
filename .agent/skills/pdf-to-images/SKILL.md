---
name: pdf-to-images
description: Converts each page of a PDF file into high-resolution PNG images. User may just say "img".
---

# PDF to Images Skill

Converts each page of both english and chinese (if there is) PDF into high-quality, high-resolution PNG images using `magick`.

## When to use this skill

- When the user wants to convert a PDF file to images for social media, presentations, or sharing.

## Workflow

1. Use `magick` with high density (DPI) to ensure high resolution.
2. Specify the output format as PNG for lossless quality.
3. The command will automatically number the pages (e.g., `output-0.png`, `output-1.png`).

Place image into new folder.

## Commands

```bash
# Convert PDF to high-quality images (300 DPI) with white background
magick -density 300 "input.pdf" -background white -alpha remove -alpha off -quality 100 "output-%d.png"
```

## Tips

- Increase `-density 600` for even higher resolution.
- Use `output-%d.png` to ensure pages are numbered correctly starting from 0.