This folder contains standalone LaTeX wrappers for previewing thesis tables.

These files do not replace the thesis table sources in `symbolic-moe/tables/`.
Each preview file simply `\input`s one table so it can be compiled on its own.

Rendered outputs should go in:

- `symbolic-moe/tables/visuals/`

Recommended workflow:

1. Edit the real table in `symbolic-moe/tables/`.
2. Compile the matching preview file in `symbolic-moe/tables/preview/`.
3. Save the generated PDF or PNG in `symbolic-moe/tables/visuals/`.
