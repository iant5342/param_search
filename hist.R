library(tidyverse)

# Settings
N_TOP <- 100
INPUT_FILE <- "results.csv"

# Load data
df <- read.csv(INPUT_FILE)

df_top <- df %>%
  arrange(desc(total_score)) %>%
  slice_head(n = N_TOP)

# Parameters to plot
params <- c(
  "theta",
  "a",
  "Delta",
  "omega_bar",
  "k",
  "lam"
)

# -----------------------------
# Plots: Full Sample vs Top Candidates
# -----------------------------
for (p in params) {
  
  g <- ggplot() +
    geom_histogram(
      data = df,
      aes(x = .data[[p]], y = after_stat(density)),
      bins = 25,
      alpha = 0.35,
      fill = "grey70",
      color = "black"
    ) +
    geom_histogram(
      data = df_top,
      aes(x = .data[[p]], y = after_stat(density)),
      bins = 25,
      alpha = 0.55,
      fill = "steelblue",
      color = "black"
    ) +
    theme_bw(base_size = 14) +
    labs(
      title = paste("Full Sample vs Top", N_TOP, "Candidates:", p),
      subtitle = "Grey = full sample; blue = top candidates by total_score",
      x = p,
      y = "Density"
    )
  
  print(g)
}

# -----------------------------
# 3. Optional: save all plots to PDF
# -----------------------------
pdf("histograms.pdf", width = 8, height = 5)

for (p in params) {
  
  g <- ggplot() +
    geom_histogram(
      data = df,
      aes(x = .data[[p]], y = after_stat(density)),
      bins = 25,
      alpha = 0.35,
      fill = "grey70",
      color = "black"
    ) +
    geom_histogram(
      data = df_top,
      aes(x = .data[[p]], y = after_stat(density)),
      bins = 25,
      alpha = 0.55,
      fill = "steelblue",
      color = "black"
    ) +
    theme_bw(base_size = 14) +
    labs(
      title = paste("Full Sample vs Top", N_TOP, "Candidates:", p),
      subtitle = "Grey = full sample; blue = top candidates by total_score",
      x = p,
      y = "Density"
    )
  
  print(g)
}

dev.off()