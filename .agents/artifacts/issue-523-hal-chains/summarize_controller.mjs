// One-off issue #523 close-out audit. Read the small controller JSON from stdin.
// Example: aws s3 cp <controller-state.json S3 URI> - --only-show-errors | node this-file.mjs
// These are controller-observed worker wall times, including pipeline overhead.
// The rate is the recorded historical node rate, not a current price or AWS invoice.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

let raw = "";
for await (const chunk of process.stdin) raw += chunk;
const data = JSON.parse(raw);
assert.equal(data.status, "succeeded");
assert.equal(data.target_species, 107);
assert.equal(data.completed_species.length, 107);
assert.equal(Object.keys(data.failed_species).length, 0);
const events = data.events.filter((event) => event.event !== "concurrency_held");
const started = events.filter((event) => event.event === "worker_started");
const completed = events.filter((event) => event.event === "worker_completed");
assert.equal(started.length, 107);
assert.equal(completed.length, 107);
assert(started.every((event) => event.attempt === 1));
const starts = new Map(started.map((event) => [event.species, Date.parse(event.time_utc)]));
assert.equal(starts.size, 107);
const rows = completed.map((event) => ({
  species: event.species,
  started_utc: started.find((start) => start.species === event.species).time_utc,
  completed_utc: event.time_utc,
  hours: (Date.parse(event.time_utc) - starts.get(event.species)) / 3_600_000,
})).sort((a, b) => a.hours - b.hours);
assert.equal(new Set(rows.map((row) => row.species)).size, 107);
assert(rows.every((row) => Number.isFinite(row.hours) && row.hours > 0));
assert.deepEqual([...starts.keys()].sort(), [...data.completed_species].sort());
const times = rows.map((row) => row.hours);
function quantile(fraction) {
  const index = fraction * (times.length - 1);
  return times[Math.floor(index)] + (index % 1) * (times[Math.ceil(index)] - times[Math.floor(index)]);
}
const start = events.find((event) => event.event === "controller_started").time_utc;
const finish = events.find((event) => event.event === "controller_finished").time_utc;
const termination = "2026-09-03T23:26:27Z";
const wall = (Date.parse(finish) - Date.parse(start)) / 3_600_000;
const idle = (Date.parse(termination) - Date.parse(finish)) / 3_600_000;
const rate = 3.63;
const result = {
  source_uri: "s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/hal-chains-directional-ramp-v2/b86897b7050bc9fdf397dd6abfb3af11fc876f86/d035c2561f6be3b11449647adfe9ce865884aef7da8b7e06b817d8a75c7f37f9/full/metadata/controller-state.json",
  source_sha256: createHash("sha256").update(raw).digest("hex"),
  pipeline_commit: data.pipeline_commit,
  config_sha256: data.config_sha256,
  controller_started_utc: start,
  controller_finished_utc: finish,
  instance_terminated_utc: termination,
  termination_evidence: "https://github.com/Open-Athena/marin-dna/issues/523#issuecomment-5533449692",
  completed_species: rows.length,
  worker_attempts: started.length,
  wall_hours: wall,
  worker_hours: {
    mean: times.reduce((a, b) => a + b, 0) / times.length,
    median: quantile(0.5),
    p90: quantile(0.9),
    p95: quantile(0.95),
    min: times[0],
    max: times.at(-1),
    quantile_method: "linear interpolation at p * (n - 1)",
  },
  recorded_node_hourly_rate_usd: rate,
  successful_recovery_compute_estimate_usd: wall * rate,
  idle_hours: idle,
  idle_compute_estimate_usd: idle * rate,
  recovery_through_termination_compute_estimate_usd: (wall + idle) * rate,
  exclusions: "Earlier pilots, staging, failed ramp, earlier idle time, storage, and requests; this is not the total session cost or an AWS billing reconciliation.",
  species: rows,
};
process.stdout.write(JSON.stringify(result, null, 2) + "\n");

