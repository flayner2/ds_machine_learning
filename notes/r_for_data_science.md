```toc
```

# R for Data Science
- Link to the book: [**R for Data Science**](https://r4ds.had.co.nz/index.html);
- Link to the solutions for the exercises on the book: [**Yet another ‘R for Data Science’ study guide**](https://brshallo.github.io/r4ds_solutions/);
- Link to the repository with the code: [**r4ds**](https://github.com/flayner2/r4ds).

This note will likely only contain important observations and snippets. The bulk of the notes and examples will be in the code in the repository linked above.

## 5. Data transformation
### 5.7 Grouped mutates (and filters)
- A grouped filter is a grouped mutate followed by an ungrouped filter, and it is generally a bad idea;
- Functions that work best for grouped mutates are called **window functions**. To learn more: `vignette("window-functions")`;
- `ifelse()` is a vectorization of the ternary operator in R.