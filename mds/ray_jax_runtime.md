# Ray + JAX Runtime Architecture

## Goal

Geodesic should be a distributed runtime for robotics and VLA experiments, not
another monolithic RL algorithm library. The core runtime should make it easy to
scale these loops:

1. collect trajectories from simulators, dataset replay, or robot workers;
2. store and sample sequence batches with clear schemas;
3. train policies with JAX on accelerators;
4. publish actor params, and eventually policy versions, back to rollout and
   evaluation workers;
5. reproduce benchmark runs across MetaWorld, LIBERO, DROID/RLDS, and VLA
   policies.

Ray owns distributed orchestration. JAX owns the numerical training path.

This design should build on the current Geodesic implementation:

- `SequenceDataset` already stores fixed-length sequence windows.
- `PrioritySampler` already sketches prioritized replay behavior.
- `ParallelReplayBuffer` already drains a Ray `Queue` into local sequence
  storage.
- `DataWorker` and `ModelSharedStorage` already provide a simple parameter
  sync mechanism for JAX actor params.

## Non-goals

- Reimplement every RL algorithm before the runtime boundary is stable.
- Put full JAX distributed training inside every Ray actor.
- Couple the first runtime design to a single VLA model implementation.
- Require robot hardware for the minimal integration tests.
- Add a second replay buffer or parameter server before the existing
  implementations are documented, tested, and extended.

## System Boundary

```
                         Geodesic runtime

    +----------------+       +-----------------------+       +-------------+
    | DataWorker     | ----> | Ray Queue             | ----> | Sequence    |
    | Ray actor      |       | transition transport  |       | replay      |
    +-------+--------+       +-----------------------+       +------+------+
            ^                                                        |
            | actor params                                           | batch
            |                                                        v
    +-------+--------+       +-----------------------+       +------+------+
    | Evaluator      | <---- | ModelSharedStorage    | <---- | JAX Learner |
    | Ray actor      |       | params + step counter |       | process     |
    +----------------+       +-----------------------+       +-------------+
```

The learner is the only component that must understand JAX train state,
optimizer state, PRNG keys, and sharding. Ray workers should treat policies as
versioned inference objects. The first implementation can keep the existing Ray
`Queue`, `ParallelReplayBuffer`, and `ModelSharedStorage` classes, then refine
their interfaces as the runtime stabilizes.

## Component Responsibilities

### DataWorker

- Runs environment interaction in Ray workers.
- Supports simulator environments first: MetaWorld and LIBERO.
- Later supports dataset replay and remote robot workers.
- Pulls the latest actor params from `ModelSharedStorage`.
- Emits trajectory fragments or complete episodes to the Ray `Queue`.
- Records metadata: environment name, task id, seed, episode id, policy version,
  success flag, and wall-clock timing.

### Sequence Replay

- Stores transition and sequence data in `SequenceDataset`.
- Accepts writes through the existing Ray `Queue` plus `ParallelReplayBuffer`
  drain thread.
- Samples batches for the learner.
- Preserves a schema that can represent state-only RL, image observations,
  language instructions, and action sequences.
- Keeps the current uniform sampling path working.
- Turns `PrioritySampler` into a tested prioritized replay path before adding
  more complex task-balanced sampling.

### JAX Learner

- Owns JAX model parameters, optimizer state, target networks, and PRNG state.
- Consumes numpy/JAX batches from sequence replay.
- Runs `jit`, `vmap`, `pmap`, or `pjit` in one controlled training process.
- Publishes actor params to `ModelSharedStorage`.
- Writes checkpoints and training metrics.

### ModelSharedStorage

- Holds the latest actor params and training step counter.
- Gives rollout and evaluator workers a stable fetch API.
- Keeps workers decoupled from learner internals.
- Can later be promoted into a more explicit policy store with named versions,
  metadata, and checkpoint-backed payloads.

### Evaluator

- Runs benchmark episodes on fixed seeds and task lists.
- Reads current or selected actor params from `ModelSharedStorage`.
- Reports success rate, return, episode length, and task-level breakdowns.
- Runs independently from training so evaluation does not block rollout.

## Data Contracts

The runtime should pass explicit batch dictionaries across component boundaries.
The first version should support these optional fields:

| Field | Shape | Required | Description |
| --- | --- | --- | --- |
| `observations` | `(B, T, ...)` | yes | State vectors or image tensors. |
| `actions` | `(B, T, A)` | yes | Continuous or discretized robot actions. |
| `rewards` | `(B, T)` | RL only | Scalar rewards. |
| `next_observations` | `(B, T, ...)` | RL only | Successor observations. |
| `dones` | `(B, T)` | RL only | Episode termination flags. |
| `instructions` | `(B,)` or token batch | VLA only | Language task prompts. |
| `task_ids` | `(B,)` or `(B, T)` | optional | MetaWorld/LIBERO task ids. |
| `metadata` | dict | optional | Seeds, source dataset, policy version. |

All component APIs should accept numpy arrays at the Ray boundary. The learner
can convert batches to JAX arrays after sampling. This avoids mixing framework
objects inside Ray serialization paths.

The current README contains an `RLDataset Schema` for state/action/reward
transitions. The missing next step is a VLA-specific schema that adds images,
language instructions, action chunks, and dataset provenance while staying
compatible with the existing sequence replay path.

## Ray and JAX Ownership

Ray should manage:

- worker lifecycles;
- rollout parallelism;
- Ray Queue transport;
- shared model storage;
- evaluator scheduling;
- cluster resource placement;
- recovery and backpressure at the orchestration level.

JAX should manage:

- model parameters and optimizer state;
- update functions;
- accelerator placement;
- `jit` compilation;
- vectorized task losses;
- multi-device sharding when the learner grows beyond one device.

The default design should keep a single learner process per experiment until
there is a measured need for multi-host training.

## Minimal Training Loop

```
initialize Ray
create ParallelReplayBuffer and Ray Queue
create ModelSharedStorage actor
create JAX learner
create N DataWorker actors
create M Evaluator actors

learner publishes initial actor params

while training:
    rollout workers fetch latest actor params
    rollout workers collect trajectory fragments
    rollout workers write fragments to Ray Queue
    replay drains queued fragments into SequenceDataset

    learner samples sequence batch from replay
    learner runs one or more JAX update steps
    learner publishes actor params to ModelSharedStorage

    evaluators periodically run fixed benchmark episodes
    metrics are logged by version, task, and seed
```

This loop should work first with a toy environment or MetaWorld smoke task. VLA
models and large image-language datasets should plug into the same boundaries
after the control loop is covered by tests.

## Policy Versioning

The current `ModelSharedStorage` stores actor params, a step counter, and a
warmstart signal. The runtime should evolve that mechanism toward explicit
policy versions.

Each published policy version should include:

- `version`: monotonically increasing integer;
- `created_at_step`: learner update step;
- `payload`: inference parameters, checkpoint path, or model reference;
- `metadata`: algorithm, model name, observation schema, action schema.

Rollout workers should write the policy version used for each episode into
trajectory metadata. This makes off-policy data provenance explicit.

## First Milestones

1. Add a VLA sequence schema for images, language instructions, action chunks,
   task ids, and dataset provenance.
2. Define runtime interface types around the existing `SequenceDataset`,
   `ParallelReplayBuffer`, `DataWorker`, `ModelSharedStorage`, learners, and
   evaluators.
3. Add lightweight tests for `ParallelReplayBuffer` queue draining and
   `PrioritySampler` behavior.
4. Document and test the existing `ModelSharedStorage` parameter sync path.
5. Add a synthetic distributed smoke test that exercises Ray Queue transport,
   replay draining, learner update, parameter publish, and evaluation.
6. Replace the synthetic environment with a MetaWorld or LIBERO smoke task.
7. Extend the VLA schema into loader utilities for RLDS, DROID, or LeRobot-style
   datasets.

## Testing Strategy

Tests should avoid heavy simulator dependencies unless they are explicitly
marked as smoke or integration tests.

- Unit tests: data schema validation, sequence replay sampling, priority
  sampling, model storage versioning.
- Runtime smoke tests: local Ray cluster with synthetic workers.
- JAX tests: deterministic update step with fixed PRNG keys.
- Integration tests: MetaWorld/LIBERO rollout behind optional markers.
- VLA tests: image-language-action batch shape and dtype checks.

The default CI path should stay lightweight and deterministic. Heavy robotics
tests can run manually or in a separate workflow once dependencies are stable.

## Design Principles

- Keep algorithm code separate from distributed orchestration.
- Prefer small typed batch contracts over implicit tuple conventions.
- Treat policy parameters as versioned artifacts.
- Keep Ray actor APIs framework-neutral.
- Move data to JAX devices inside the learner, not inside the replay actor.
- Make the minimal runtime work before adding large VLA models.
- Extend existing replay and parameter-sync code before introducing replacement
  components.
