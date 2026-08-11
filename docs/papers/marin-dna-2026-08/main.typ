#import "template.typ": paper

#show: paper.with(
  title: [Data curation and hyperparameter transfer enable competitive variant-effect prediction with a 1B genomic language model],
  authors: [TODO],
  author-metadata: ("TODO",),
  date: [August 2026],
  abstract: [#include "sections/abstract.typ"],
)

#include "sections/introduction.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"
#include "sections/methods.typ"
#include "sections/availability.typ"
#include "sections/provenance.typ"
#include "sections/acknowledgements.typ"
#include "sections/statements.typ"

#bibliography("references.bib", title: "References", style: "apa")
#include "sections/supplement.typ"
