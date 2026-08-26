suppressPackageStartupMessages(library(Biostrings))
suppressPackageStartupMessages(library(DECIPHER))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 13) {
  stop("expected 13 arguments")
}

input_fasta <- args[[1]]
output_assignments <- args[[2]]
output_version <- args[[3]]
cutoff <- as.numeric(args[[4]])
min_coverage <- as.numeric(args[[5]])
random_seed <- as.integer(args[[6]])
processors <- as.integer(args[[7]])
max_phase1 <- as.integer(args[[8]])
max_phase2 <- as.integer(args[[9]])
max_phase3 <- as.integer(args[[10]])
max_alignments <- as.integer(args[[11]])
rare_kmers <- as.integer(args[[12]])
probability <- as.numeric(args[[13]])

sequences <- readDNAStringSet(input_fasta, use.names = TRUE)
sequence_ids <- names(sequences)
if (is.null(sequence_ids) || anyNA(sequence_ids) || any(sequence_ids == "")) {
  stop("every FASTA record must have a non-empty identifier")
}
if (anyDuplicated(sequence_ids)) {
  stop("FASTA identifiers must be unique")
}

set.seed(random_seed)
clusters <- Clusterize(
  sequences,
  cutoff = cutoff,
  minCoverage = min_coverage,
  maxPhase1 = max_phase1,
  maxPhase2 = max_phase2,
  maxPhase3 = max_phase3,
  maxAlignments = max_alignments,
  rareKmers = rare_kmers,
  probability = probability,
  invertCenters = TRUE,
  singleLinkage = FALSE,
  maskRepeats = TRUE,
  maskLCRs = TRUE,
  processors = processors,
  verbose = TRUE
)

raw_cluster_ids <- as.integer(clusters[[1]])
if (length(raw_cluster_ids) != length(sequence_ids)) {
  stop("Clusterize returned the wrong number of assignments")
}
if (anyNA(raw_cluster_ids) || any(raw_cluster_ids == 0L)) {
  stop("Clusterize returned an invalid cluster identifier")
}

cluster_ids <- abs(raw_cluster_ids)
center_indices <- which(raw_cluster_ids < 0L)
centers_by_cluster <- split(sequence_ids[center_indices], cluster_ids[center_indices])
representatives_by_cluster <- vapply(
  centers_by_cluster,
  function(ids) sort(ids, method = "radix")[[1]],
  character(1)
)
if (!setequal(names(representatives_by_cluster), as.character(unique(cluster_ids)))) {
  stop("Clusterize did not identify a center for every cluster")
}

assignments <- data.frame(
  representative = unname(representatives_by_cluster[as.character(cluster_ids)]),
  member = sequence_ids,
  stringsAsFactors = FALSE
)
write.table(
  assignments,
  file = output_assignments,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)
writeLines(
  sprintf("DECIPHER %s; R %s", packageVersion("DECIPHER"), getRversion()),
  output_version
)
