files = list.files("/home/dc/vscode/vscode20260406/rgcnformer_sum/ipynb", pattern = "\\.R\$", full.names = TRUE, recursive = TRUE)
for (f in files) {
  r = tryCatch({ parse(f); "OK" }, error = function(e) paste("ERR", e$message))
  cat(r, " ", f, "
")
}
