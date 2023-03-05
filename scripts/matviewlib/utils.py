from typing import Union, List, Iterable

def ensure_install(module_name: str, lib_name: Union[str,None] = None):
    import sys, traceback
    from importlib.util import find_spec
    
    if lib_name is None:
        lib_name = module_name
    
    if find_spec(module_name) is None:
        import subprocess
        try:
            print('-' * 80, file=sys.stderr)
            print(f'| installing {lib_name} ...', file=sys.stderr)
            print('-' * 80, file=sys.stderr)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", lib_name],
                stdout=sys.stdout,
                stderr=sys.stderr
            )
        except Exception as e:
            msg = ''.join(traceback.TracebackException.from_exception(e).format())
            print(msg, file=sys.stderr)
            print('-' * 80, file=sys.stderr)
            print(f'| failed to install {lib_name}. exit...', file=sys.stderr)
            print('-' * 80, file=sys.stderr)
            sys.exit(1)


def try_to_int(v: str) -> Union[str,int]:
    try:
        return int(v)
    except ValueError:
        return v

def match(piece: str, list: List[str]):
    return any(piece in s for s in list)

def match_any(pieces: Iterable[str], list: List[str]):
    return any(match(p, list) for p in pieces)
