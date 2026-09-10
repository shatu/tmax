# Runtime requirements

`tblite-clean` fixes task/bootstrap defects; it does not make unsupported
container features available on a cluster. Preserve the task CPU, memory and
timeout settings. Record nominal disk allowance separately from measured space.

## ETL PostgreSQL user switching

`etl_checkpoint_resume_bug` starts PostgreSQL as the `postgres` user. Its image
startup fix selects the TCP endpoint; that is separate from user switching.

- A Docker runtime that supports the image's users passes the repaired task.
- Apptainer needs subordinate UID **and** GID ranges and functioning mappings
  so `su postgres` can actually become the non-root user. Oscar independently
  confirmed this capability on his workers (private issue #1, comment 5614044936).
- A root-only mapping is insufficient. Hyak job 39936627 reproduced PostgreSQL
  refusing to run as root; empty `/etc/subuid` and `/etc/subgid` were observed
  on our Hyak and Tillicum logins. Do not count this as a model reward zero.

Cluster administrators control those mappings. Do not spoof identity checks,
silently skip the task, or pool this infrastructure failure with scored trials.
The dataset retains ETL; acceptance on a supported runtime is distinct from
availability on a particular site. ACL remains the explicitly excluded task.

## React filesystem performance

The original verifier removes `node_modules` before reinstalling dependencies.
This takes substantially longer on a FUSE-backed disk overlay than on local
Docker. The repaired task nevertheless passed at its original resource/time
limits on Hyak (job 39937635). Keep that result separate from directory-backed
runs that waive disk enforcement; do not silently claim filesystem parity.
