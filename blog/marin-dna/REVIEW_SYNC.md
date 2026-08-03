# Website and Google Doc review sync

This is the operator runbook for
[issue #408](https://github.com/Open-Athena/marin-dna/issues/408). It applies
only to the permanent, unmerged `claude/issue-373-blog-staging` branch. A sync
is an explicit user-requested snapshot, never a watcher or an automatic
reaction to a push.

The Markdown article remains authoritative. Every distinct sync request updates
the existing website preview branch and creates a **new, immutable Google Doc**
for editorial review. Retrying the same request recovers or reuses its existing
artifacts; it must not create another Doc.

## First-sync Drive gate

Do not perform the first external sync until the user has chosen and the
operator has recorded all three values below in the private sync record. Mirror only the non-sensitive decision status in issue #408:

| Decision | Required value |
| --- | --- |
| Destination folder | Google Drive folder URL and immutable folder ID |
| Owner | Account or shared-drive owner that will own every review Doc |
| Sharing policy | Exact reader, commenter, or writer audience |

The sync must use the connected Google Drive account and repository
authentication. Never write credentials, refresh tokens, cookies, or API keys
into this repository or a generated bundle.

The older
[June 25 review Doc](https://docs.google.com/document/d/17fiMIAUq2Wr-ZZHob6T2rO_2hQrkwUwG5TMv2Y7i-ZE/edit)
is a structural reference only. Do not copy its content, comments, sharing
settings, or stale facts into a new snapshot.

## Why the transport is DOCX

Google Docs has an
[official Markdown import](https://support.google.com/docs/answer/12014036)
for basic headings, emphasis, and links. The canonical article also contains
local SVG figures, captions, frontmatter, website-only `<details>` blocks,
inline HTML, and Markdown footnotes. Those constructs do not survive the basic
import as a complete review document.

The preparer therefore emits both:

- `review.md`, an inspectable normalized Markdown companion; and
- `review.docx`, a self-contained import transport with native headings,
  centered title/author/date/summary, hyperlinks, native footnotes, the 19
  figures converted to embedded high-resolution PNGs, captions in article order, and a
  prominent commit-pinned provenance notice.

The DOCX is only a transport. Neither it nor the resulting Google Doc becomes a
second source of truth.

## Stable request identity and durable record

Choose one request ID before any external effect. Recommended form:

```text
issue-408-YYYYMMDDTHHMMSSZ-<source-sha-prefix>
```

The request identity is the tuple:

```text
(request_id, source_sha, requested_at, bundle_sha256)
```

Keep the canonical append-only JSONL ledger in the approved
organization-owned Drive folder alongside the review Docs. Before a sync,
download the current ledger to a private ignored workspace path; after each
appended event, replace the Drive copy only after confirming its Drive file
version has not changed. Never commit the ledger because Doc IDs, URLs, and
revision IDs are organization-only review metadata.

Issue #408 comments are a redacted human-visible mirror, not the retry ledger.
They may record the request ID, source and website commits, document title, and
per-target status, but must omit the Drive folder ID, Doc ID, Doc URL, and
revision ID unless the issue access policy has been verified to allow them.
Each comment begins with `🤖`; do not edit or delete prior event comments.

The private event schema implemented by `blog_review_sync.SyncEvent` records:

```json
{
  "request_id": "issue-408-...",
  "source_sha": "40 lowercase hex characters",
  "target": "request | website | document",
  "status": "registered | succeeded | failed",
  "recorded_at": "YYYY-MM-DDTHH:MM:SSZ",
  "details": {}
}
```

Register the request before the first external mutation. A website success
records `commit_sha`, `preview_url`, and `build_status`. A document success
records `document_url`, `revision_id`, and `verification_status`. A failure
records an `error` string. On retry, read all private-ledger events for the
request ID and run only targets without a verified success receipt.

Using a request ID with different identity fields is a hard conflict. Two
different request IDs intentionally create two review Docs even when they point
to the same source SHA.

## 1. Resolve and prepare the exact source

Fetch the remote staging tip and work from that exact commit:

```bash
git fetch origin claude/issue-373-blog-staging
git switch claude/issue-373-blog-staging
git merge --ff-only origin/claude/issue-373-blog-staging
git rev-parse HEAD
```

Do not substitute `main`, a child editing branch, an uncommitted worktree, or a
previously remembered SHA. Confirm the worktree is clean before the sync.

Prepare the ignored immutable bundle:

```bash
uv run python src/marin_dna/blog_review_sync.py prepare \
  --source-sha <40-character-source-sha> \
  --request-id <stable-request-id> \
  --requested-at <YYYY-MM-DDTHH:MM:SSZ> \
  --destination \
  blog/marin-dna/export/review-sync/<stable-request-id>
```

The command validates all local assets and footnotes, preserves the 19 figures
in source order, converts SVGs, renders native Word footnotes into the DOCX,
normalizes prose soft-wraps to spaces, hashes every generated file,
and writes `manifest.json` atomically. Repeating the command with the same
identity verifies and reuses the existing bundle. Any changed file or identity
field fails loudly.

Append the request-registration event, including the printed manifest SHA-256,
to the private ledger before continuing, upload it with a Drive file-version
conflict guard, and mirror only the redacted request status in issue #408.

## 2. Update the existing website preview

Use the existing Open Athena website PR #59 head branch
`cms/blog/genomic-lm-optimization`; do not open a replacement PR.

1. Fetch and fast-forward a clean checkout of
   `Open-Athena/open-athena.github.io`.
2. Export from the exact source checkout:

   ```bash
   uv run --no-project python src/marin_dna/blog_workspace.py export \
     --destination /absolute/path/to/open-athena.github.io
   ```

3. Assert that the only changed paths are:

   ```text
   content/blog/marin-dna.md
   static/assets/images/blog/marin-dna/**
   ```

   Preserve every unrelated PR change. Never force-push.
4. Run `uv run build.py` in the website checkout.
5. Commit the exact source SHA and request ID in the commit message, then push
   the existing branch normally.
6. Wait for the Cloudflare preview deployment to finish. Verify that
   [the stable preview URL](https://cms-blog-genomic-lm-optimiza.openathena-ai.pages.dev/blog/marin-dna/)
   serves the new commit.
7. Append either a website success or failure event to the private ledger
   immediately, upload it with a Drive file-version conflict guard, and mirror
   the non-sensitive status in issue #408.

If the exact article/assets are already present, the website adapter may reuse
the existing verified website commit. It must still report that commit and the
successful preview status.

## 3. Create the immutable Google Doc

Import `review.docx` through Google Drive's native conversion flow using the
bundle's `document_title`. Move it to the approved folder and apply exactly the
approved sharing policy. Keep `documentFormat.documentMode = PAGES` so native
footnotes render at the bottom of their reference page; verify the readback
reports `PAGES`. Switch to `PAGELESS` only when the user explicitly requests it
for that snapshot.

Before recording success, verify:

- centered title, authors, date, and summary;
- the blue provenance notice links to the exact source SHA and names the
  request ID, generation time, authoritative staging branch, and review-only
  status of Doc edits;
- native Google Docs heading structure and paged document mode;
- all 19 figures in canonical order, each followed by its caption;
- working article hyperlinks and all 27 Markdown notes as native, hoverable
  Google Docs footnotes in first-reference order; and
- the Drive folder, owner, and permissions match the first-sync decision.

Record the Google Doc URL, its current revision ID, and verification success in
the private ledger, upload it with a Drive file-version conflict guard, and add
a redacted success event to issue #408. Never edit, replace, or delete a Doc
recorded as a successful artifact. A later user sync receives a new request ID and a new Doc.

Comment extraction is not part of this export. It is a separate read workflow
used only when an agent needs to inspect review feedback in an existing Doc.

## 4. Recovery and final report

The website and document effects are independent. Attempt both even if one
fails. On a retry:

- verified website + missing Doc → create or recover only the Doc;
- verified Doc + missing website → update or recover only the website;
- both verified → report the existing receipts and perform no writes;
- conflicting identity or two different successful Docs for one request → stop
  for operator review.

Report these fields to the user after every attempt:

```text
request ID
source SHA
website commit SHA, preview URL, build/deployment status
Google Doc URL, revision ID, verification status
any partial failure and the exact retry action remaining
```
