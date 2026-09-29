from os import name
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Any

# check if running on windows
WIN32 = name == "nt"
ST_INFO = None
# Explicitly widened: the Windows branch holds a STARTUPINFO, the POSIX branch an
# empty dict, so the inferred type is a platform-dependent union. `Any` is what
# lets the `**` splat below reach Popen's overloads at all -- typeshed declares
# them for concrete keyword types, and a splatted union matches none of them.
subprocess_flags: dict[str, Any]
if WIN32:
    import ctypes
    from subprocess import (
        CREATE_NO_WINDOW,
        REALTIME_PRIORITY_CLASS,
        STARTF_USESHOWWINDOW,
        STARTF_USESTDHANDLES,
        STARTUPINFO,
        SW_HIDE,
    )

    ST_INFO = STARTUPINFO()
    ST_INFO.dwFlags |= STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | REALTIME_PRIORITY_CLASS
    ST_INFO.wShowWindow = SW_HIDE
    CR_FLAGS = CREATE_NO_WINDOW
    subprocess_flags = {
        "startupinfo": ST_INFO,
        "creationflags": CR_FLAGS,
        "start_new_session": True,
    }
else:
    subprocess_flags = {}


def _terminate_process(
    process: Popen[str],
) -> None:
    """Terminate a process forcefully on Windows."""
    handle = ctypes.windll.kernel32.OpenProcess(1, False, process.pid)
    ctypes.windll.kernel32.TerminateProcess(handle, -1)
    ctypes.windll.kernel32.CloseHandle(handle)


def run(
    args: list[str],
) -> tuple[int, str]:
    with Popen(
        args,
        shell=False,
        bufsize=-1,
        stdout=PIPE,
        stderr=PIPE,
        close_fds=True,
        # `text=True` is the current spelling of `universal_newlines=True`;
        # typeshed's universal_newlines overload additionally demands `encoding`,
        # which left no matching overload for this call.
        text=True,
        **subprocess_flags,
    ) as process:
        try:
            result, _ = process.communicate(timeout=5)
        except TimeoutExpired:
            if WIN32:
                # pylint: disable=protected-access
                _terminate_process(process)
            process.kill()
            result, _ = process.communicate()
            raise

        return (process.returncode, result)
