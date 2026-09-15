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
    ):
        if old in text:
            text = text.replace(old, new, 1)
            applied.append(name)

    if applied:
        MODAL_PY.write_text(text)
        print(f"patched modal.py: {', '.join(applied)} -> Sandbox.filesystem API")
    else:
        print("modal.py: already on the Sandbox.filesystem API (no changes)")

    leftover = [c for c in ("_sandbox.ls.aio", "_sandbox.mkdir.aio") if c in text]
    if leftover:
        raise SystemExit(
            "patch_harbor_modal: legacy filesystem calls remain: "
            f"{', '.join(leftover)}. harbor's modal.py has changed shape -- "
            "update this patch."
        )


if __name__ == "__main__":
    main()
