# Research question documents

Documents in this directory are MarinDNA's accepted research synthesis. A question document on `main` records the current answer, confidence, limitations, operational consequence, and unresolved work. Experiment issues and logbooks may contain newer evidence that has not yet been synthesized.

Use [`_template.md`](_template.md) for a new question and add it to [`index.md`](index.md). Question files use `rq-NNNN-short-slug.md`; the `RQ-NNNN` identity does not change when the title, scope, or status changes.

## Workflow

- Humans declare or approve question scope.
- Agents may propose documents and revisions through normal pull requests.
- `main` is the accepted synthesis. Branches and open pull requests are proposals.
- Open a synthesis pull request only when evidence changes the answer, confidence, an important limitation, operational consequence, scope, relationships, status, or successor.
- Keep routine progress, detailed results, failures, and one-off analysis in experiment issues, logbooks, permanent branches, W&B, or another artifact store.

Git history and pull requests are the review record. This first version does not add CODEOWNERS or separate approval automation.

## Schema

Keep these sections in order:

1. `Metadata`
2. `Question and scope`
3. `Current answer`
4. `Confidence and limitations`
5. `Operational consequence`
6. `Supporting evidence`
7. `Contradictory evidence`
8. `Related experiments`
9. `Open questions`
10. `History`

The metadata table requires:

- `Question ID`: unique `RQ-NNNN` identity consistent with the filename.
- `Status`: `active`, `superseded`, or `closed`.
- `Overall confidence`: `low`, `medium`, `high`, or `unknown`.
- `Evidence considered through`: ISO date through which the evidence was assessed.
- `Predecessor issues`: legacy research-question links, or `None`.

Use commit-pinned repository links where reproducibility matters. Separate reported source findings from MarinDNA interpretation and untested hypotheses. Keep supporting and contradictory evidence explicit.

`Related experiments` is exhaustive. Each item links an `experiment` issue and states its current contribution. Each experiment issue links back to the canonical document on `main`. Add or remove a relationship through the document's normal pull-request workflow; do not use GitHub sub-issue metadata.

## Lifecycle

### Create

1. Obtain human approval for the question and scope.
2. Search active, superseded, and closed documents for overlap.
3. Copy the template, allocate a unique ID, and make unknowns explicit.
4. Add the document to the index and open a pull request.

### Update

Read the accepted document, pending synthesis pull requests, linked experiments and comments, logbooks, permanent branches, source ledgers, and relevant literature. Update the synthesis only when its answer or implications materially change.

### Supersede or close

Set `Status` to `superseded` or `closed`, explain the decision and successor under `History`, and update the index. Do not delete the document or reuse its ID.

### Handle pending evidence

Leave the accepted document unchanged while evidence is pending. Record the evidence in its experiment issue and logbook. The open pull request is the proposed synthesis until it merges.

## Legacy issue migration

Legacy issues remain historical records. Migrate each current body into a document and preserve the predecessor link. After the document merges:

1. Prepend an archival notice linking the canonical document and migration pull request. Preserve the previous body below it.
2. Post one final `🤖` comment linking the document and pull request.
3. Close formerly open research-question issues. Add the same notice to already closed issues without reopening them.
4. Keep the historical label if useful for discovery. Do not create a replacement tracker issue.

The initial workflow uses GitHub-rendered Markdown and has no documentation hosting dependency.
