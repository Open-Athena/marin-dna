# Unattended Codex Work

Preserve task approval across sessions and check execution access before starting work that must continue without the user present.
The agent follows [Task Authorization](../../AGENTS.md#task-authorization); the user or host owner controls the session's execution permissions.
These instructions apply to future tasks that load this repository's guidance.
Merging them does not change an existing session's permission settings.

## Initial Preflight And Resumption

1. Recover the task's existing authorization from available user instructions and private task context.
   Record its source and date, permitted actions and resource types, cumulative spending cap, credential purposes and destinations, publication scope, and recovery limits.
   Carry forward spending from failed attempts and previous sessions.
   Keep original-message references in private handoff context; public logbooks can record scope and limits without quoting private conversation text.
2. Inspect the current session's effective permission profile or sandbox, approval policy, reviewer, and any managed restrictions.
   Inspect only relevant configuration keys; do not dump configuration files or credential stores.
   A config file expresses a requested setting; the running client's effective settings determine access.
3. Check the access needed by the remaining workflow, including launch, monitoring, runtime extensions, recovery, artifact upload, and cleanup.
   Use read-only service checks and reversible local probes where useful, following the repository's credential instructions.
   Verify credential availability and destination access without printing secret values.
   A successful read or one approved command does not establish permission for later writes or paid operations.
4. Ask once at initial preflight for missing task authority, material decisions, or a concrete client/host setting change.
   When authority already exists, explain the execution restriction and its remedy without asking the user to approve the task again.
   Continue independent work while a dependent operation is blocked.
   On resumption, repeat access checks for the current environment and reconcile the authorization record with available user instructions.

Use this compact handoff record, omitting fields that do not apply:

| Field | Record |
|---|---|
| Authorization | Source/date, standing instructions, approved scope, unresolved decisions; private reference to original approval |
| Budget | Cumulative cap, spent/committed amount, remaining amount, resource types, recovery allowance |
| Credentials and publication | Credential purpose, service/account, destination and artifact class; no credential values |
| Execution access | Effective profile/sandbox, approval policy/reviewer, managed restrictions, probe results and unresolved restrictions |
| Active resources | Provider IDs, job IDs, shutdown times, durable artifact locations, resume and cleanup commands |

## Owner-Controlled Full Access

Use this setup when the user has requested unrestricted local execution for authorized work.
Full Access removes local sandbox restrictions; task scope, spending limits, service permissions, and separate connector controls still apply.
Managed policy can restrict the available modes. [OpenAI permission profiles](https://learn.chatgpt.com/docs/permissions)

### Desktop App Or IDE

Select **Full access** using the permissions control below the composer.
If the option is hidden, the documented desktop setup enables its visibility under **Settings > General > Permissions**.
Enabling the option only adds it to the menu; select it for the task and verify the active mode.
Check the mode again for a new or resumed task, because making an option visible does not change existing tasks. [OpenAI permission modes](https://learn.chatgpt.com/docs/permission-modes)

### Persistent CLI Profile

For a reusable, explicitly selected CLI configuration, the owner can save the following as `~/.codex/marin-unattended.config.toml` (or in the configured `CODEX_HOME` directory):

```toml
approval_policy = "never"
default_permissions = ":danger-full-access"
```

Select that configuration when starting or resuming authorized work:

```bash
codex --profile marin-unattended --cd /path/to/marin-dna
codex resume --profile marin-unattended --cd /path/to/marin-dna SESSION_ID
```

This named configuration persists across CLI sessions when selected with `--profile`.
To make these settings the user-wide CLI default instead, the owner can place the same top-level keys in their existing `~/.codex/config.toml`, preserving unrelated settings.
That default applies beyond this repository.
Current CLI profiles use separate `<name>.config.toml` files; the older `[profiles.<name>]` format is unsupported in Codex 0.134.0 and later. [OpenAI configuration profiles](https://learn.chatgpt.com/docs/config-file/config-advanced#profiles)

Do not mix `default_permissions` with legacy `sandbox_mode`, `[sandbox_workspace_write]`, or `--sandbox` settings.
Remove conflicting legacy settings from the owner's configuration when adopting permission profiles.
Project settings and CLI overrides can supersede user profiles, and managed requirements can forbid Full Access or `approval_policy = "never"`.
Verify the effective settings after startup. [OpenAI configuration precedence](https://learn.chatgpt.com/docs/config-file/config-basic#configuration-precedence)

The profile format and CLI start/resume flags were checked against Codex CLI 0.153.2 and official documentation on 2026-09-10.
Check the installed client's help and current official documentation when those interfaces change.
These are owner setup instructions; an agent in a restricted session must not use them to start an unrestricted child or rewrite enforced policy after a rejection.

## Recognize Incomplete Setup

| Observed setting or failure | Meaning and next action |
|---|---|
| Workspace restrictions with **Approve for me** / `auto_review` | Automatic review handles escalation requests and can reject them. Check whether required operations remain subject to review and identify the setup mismatch before promising unattended execution. |
| `approval_policy = "never"` with a restrictive sandbox | Prompts are disabled while access remains restricted. The owner must select permissions that allow the required operations. |
| Full Access is unavailable or effective settings remain restricted | Ask the client/host owner to resolve the specific managed restriction through supported settings. Preserve the existing task approval. |
| A service reports missing credentials or insufficient rights | Diagnose that service's access using its normal client. Local Full Access cannot grant provider-side rights. |
| An already authorized action is rejected | Check the command and approval source, supply that evidence when supported, or use an authorized alternative. Report any remaining platform blocker precisely; do not repeat unchanged rejected requests. |

Automatic review changes who handles approvals while retaining the sandbox boundary.
It provides no guarantee that a later operation will be approved. [OpenAI automatic review](https://learn.chatgpt.com/docs/sandboxing/auto-review)

## Worker Deadlines And Recovery

Before launch, estimate the complete path through setup, computation, final assembly, export, and durable upload.
Set worker lifetimes and recovery headroom within the approved cumulative budget, including coordinator and failed-attempt costs.
Arrange durable checkpoints or intermediate outputs before ephemeral storage can disappear.

Compare observed throughput with time remaining until shutdown early enough to extend, checkpoint, or replace the worker under the existing authorization.
Do not make a predictable timeout depend on the user returning to repeat that authorization.
Keep budget and shutdown limits in force; request new authority only for an actual expansion beyond them.
If execution restrictions block recovery, save accessible outputs and release idle paid resources where authorized, record the exact blocker and resume path, and continue unaffected work.
