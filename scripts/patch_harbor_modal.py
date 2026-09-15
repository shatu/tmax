"""Port harbor 0.6.6's Modal environment onto Modal's current filesystem API.

harbor 0.6.6 predates Modal's removal of the legacy Sandbox filesystem calls
(``Sandbox.mkdir`` / ``Sandbox.ls``). Modal dropped them SERVER-side, so pinning
an older SDK does not help -- every trial dies in ``_setup_environment`` with::

    ConflictError: The legacy Sandbox filesystem API is no longer supported.

and the job reports 0 trials / N exceptions with a meaningless 0.0 mean.

Three call sites need rewriting onto ``Sandbox.filesystem.*``, which harbor
already uses for upload/download:

  * ``sandbox.mkdir(p, parents=True)`` -> ``filesystem.make_directory(p, create_parents=True)``
  * ``is_dir`` / ``is_file``, which inferred type from which exception
    ``sandbox.ls`` raised, -> ``filesystem.stat(p)`` + ``FileInfo.is_dir()`` /
    ``.is_file()``. Note the miss case raises Modal's own
    ``SandboxFilesystemNotFoundError``, which is NOT a ``FileNotFoundError``
    subclass, so the except clause has to name it.

It also makes command output decoding lenient. harbor reads Modal's process
streams in TEXT mode, and Modal decodes them as strict UTF-8, so a single
non-UTF-8 byte anywhere in a command's stdout raises ``UnicodeDecodeError``
inside ``_sdk_exec`` and kills the whole TRIAL -- not just that command. Tasks
that cat a binary, hexdump, or crypto output hit this routinely: it errored 5 of
89 trials on terminal-bench@2.0 (git-multibranch, vulnerable-secret,
password-recovery, financial-document-processor, feal-linear-cryptanalysis),
silently depressing pass@1. Reading with ``text=False`` and decoding with
``errors="replace"`` matches what a real terminal does with stray bytes.

Idempotent: safe to re-run, and a no-op on a harbor that no longer has the
legacy calls.
"""

import pathlib

import harbor

MODAL_PY = pathlib.Path(harbor.__file__).parent / "environments/modal.py"

LEGACY_IS_DIR = '''        try:
            await self._env._sandbox.ls.aio(path)
            return True
        except (NotADirectoryError, FileNotFoundError):
            return False'''

NEW_IS_DIR = '''        from modal.exception import SandboxFilesystemNotFoundError

        try:
            info = await self._env._sandbox.filesystem.stat.aio(path)
            return info.is_dir()
        except SandboxFilesystemNotFoundError:
            return False'''

LEGACY_IS_FILE = '''        try:
            await self._env._sandbox.ls.aio(path)
            return False
        except NotADirectoryError:
            return True
        except FileNotFoundError:
            return False'''

NEW_IS_FILE = '''        from modal.exception import SandboxFilesystemNotFoundError

        try:
            info = await self._env._sandbox.filesystem.stat.aio(path)
            return info.is_file()
        except SandboxFilesystemNotFoundError:
            return False'''

LEGACY_MKDIR = '''        await env._sandbox.mkdir.aio(str(EnvironmentPaths.agent_dir), parents=True)
        await env._sandbox.mkdir.aio(str(EnvironmentPaths.verifier_dir), parents=True)'''

NEW_MKDIR = '''        await env._sandbox.filesystem.make_directory.aio(
            str(EnvironmentPaths.agent_dir), create_parents=True
        )
        await env._sandbox.filesystem.make_directory.aio(
            str(EnvironmentPaths.verifier_dir), create_parents=True
        )'''


STRICT_DECODE = """        process = await self._sandbox.exec.aio(
            shell,
            "-lc" if login else "-c",
            command,
            workdir=cwd,
            secrets=[Secret.from_dict(env)] if env else [],  # type: ignore
            timeout=timeout_sec,
        )

        stdout = await process.stdout.read.aio()
        stderr = await process.stderr.read.aio()"""

LENIENT_DECODE = """        process = await self._sandbox.exec.aio(
            shell,
            "-lc" if login else "-c",
            command,
            workdir=cwd,
            secrets=[Secret.from_dict(env)] if env else [],  # type: ignore
            timeout=timeout_sec,
            text=False,
        )

        # Bytes, not str: Modal decodes text-mode streams as strict UTF-8, so one
        # stray byte from a task that cats a binary would raise UnicodeDecodeError
        # here and fail the whole trial. Replace undecodable bytes instead.
        stdout = (await process.stdout.read.aio()).decode("utf-8", errors="replace")
        stderr = (await process.stderr.read.aio()).decode("utf-8", errors="replace")"""


def main() -> None:
    if not MODAL_PY.exists():
        print(f"patch_harbor_modal: {MODAL_PY} not found; nothing to do")
        return

    text = MODAL_PY.read_text()
    applied = []
    for name, old, new in (
        ("is_dir", LEGACY_IS_DIR, NEW_IS_DIR),
        ("is_file", LEGACY_IS_FILE, NEW_IS_FILE),
        ("mkdir", LEGACY_MKDIR, NEW_MKDIR),
        ("lenient-decode", STRICT_DECODE, LENIENT_DECODE),
    ):
        if old in text:
            text = text.replace(old, new, 1)
            applied.append(name)

    if applied:
        MODAL_PY.write_text(text)
        print(f"patched modal.py: {', '.join(applied)}")
    else:
        print("modal.py: already patched (no changes)")

    if "text=False" not in text:
        raise SystemExit(
            "patch_harbor_modal: exec still reads streams in text mode. harbor's "
            "modal.py has changed shape -- update this patch."
        )

    leftover = [c for c in ("_sandbox.ls.aio", "_sandbox.mkdir.aio") if c in text]
    if leftover:
        raise SystemExit(
            "patch_harbor_modal: legacy filesystem calls remain: "
            f"{', '.join(leftover)}. harbor's modal.py has changed shape -- "
            "update this patch."
        )


if __name__ == "__main__":
    main()
