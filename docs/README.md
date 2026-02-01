# Diffome Documentation

This directory contains documentation for the Diffome project.

## Journal Article

The file `diffome_article.tex` contains a draft journal article manuscript suitable for submission to top NeuroAI journals such as:

- **Nature Neuroscience**
- **Neuron**
- **Neural Computation**
- **Nature Machine Intelligence**

### Compiling the LaTeX Document

To compile the LaTeX document into a PDF:

```bash
# Standard compilation
pdflatex diffome_article.tex
bibtex diffome_article
pdflatex diffome_article.tex
pdflatex diffome_article.tex
```

Or use latexmk for automatic compilation:

```bash
latexmk -pdf diffome_article.tex
```

### Required LaTeX Packages

The document requires the following LaTeX packages:
- `inputenc`, `fontenc` - Text encoding
- `amsmath`, `amssymb`, `amsthm` - Mathematical typesetting
- `graphicx` - Figure inclusion
- `geometry` - Page layout
- `natbib` - Bibliography management
- `hyperref` - Hyperlinks
- `algorithm`, `algorithmic` - Algorithm pseudocode
- `subcaption` - Subfigures
- `booktabs` - Professional tables

Most standard LaTeX distributions (TeX Live, MiKTeX) include all these packages by default.

### Article Structure

The article includes:

1. **Abstract** - Overview of Diffome and its applications
2. **Introduction** - Background on connectomics and topological data analysis
3. **Methods** - Technical details of the framework
   - Connectome construction from diffusion MRI
   - Topological feature extraction via persistent homology
   - Connectome comparison metrics
   - Statistical inference methods
   - Software implementation
4. **Results** - Validation and applications
   - Synthetic data validation
   - Clinical applications (DBS outcomes)
   - Comparison with graph-theoretic approaches
   - Multi-scale analysis
5. **Discussion** - Implications and future directions
6. **Conclusion** - Summary and impact
7. **References** - Comprehensive bibliography

### Customization

To customize the article for your submission:

1. Update author information in the preamble
2. Add institutional affiliations
3. Include actual results and figures from your analyses
4. Adjust the bibliography style as needed for target journal
5. Add acknowledgments and funding information
6. Update data availability statements

### Online Compilation

If you don't have LaTeX installed locally, you can use online services:

- [Overleaf](https://www.overleaf.com/) - Upload the .tex file and compile online
- [Papeeria](https://papeeria.com/) - Another online LaTeX editor
- [LaTeX.Online](https://latexonline.cc/) - Web-based compilation service

### License

This documentation is part of the Diffome project. See the main repository LICENSE file for details.
