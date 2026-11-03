# Reports and presentation

- [IEEE report](reports/ieee.pdf)
- [Springer report](reports/springer.pdf)
- [Editable PowerPoint](presentation/parallel_monte_carlo.pptx)
- [Slide PDF](presentation/parallel_monte_carlo.pdf)
- [Speaker notes](presentation/speaker_notes.md)
- [Live demo screenshot](demo/live_demo.png)
- [Hover explanation screenshot](demo/explanation_tooltip.png)
- [Measurements](data/README.md)

The reports share one technical body and bibliography, with format-specific
figure sizes. Results describe one laptop and include the stated RNG and timing
limitations. See the repository documentation for the methodology and literature.

## Rebuild reports

Use Tectonic from `artifacts/reports`:

```text
tectonic ieee.tex
tectonic springer.tex
```

Alternatively use a TeX installation with IEEEtran and LLNCS, compiling each
wrapper with LaTeX, BibTeX, and two more LaTeX passes.

## Rebuild slides

From `artifacts/presentation`, run `npm install` and `npm run build`.
The editable deck contains speaker notes for all 17 slides. Export it to PDF
from PowerPoint or another compatible presentation application after changes.
