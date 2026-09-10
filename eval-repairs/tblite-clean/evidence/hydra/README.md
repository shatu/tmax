# Hydra full Harbor acceptance

These are verbatim result, image/resource, verifier, and pool-cleanup files from
Hyak driver 39937179 and trial `hydra-debug-slurm-mode__JqgddAN`.
Source task: f075e463472c7790b85793b392dff1fff20cc0e3.
Packaging recipe: b299b4b. Sandfleet source: 95901e6 (merged as db1dd89).

Observed result: reward 1.0, no exception, all seven assertions pass. The original
task limits and assertions were retained. These oracle results are not model
checkpoint evaluations and must not be added to the checkpoint score table.

Scheduler accounting independently showed driver 39937179 COMPLETED 0:0 and
worker 39937232 CANCELLED, with batch/extern COMPLETED 0:0 and worker step
CANCELLED 0:15. The controller deleted its pool. Temporary credentials were
removed after both jobs were terminal; credentials are not included here.

Raw receipts contain private cluster paths. Keep these in the private repository;
they are not part of a public dataset distribution.
