# error: 경로가 올바르지 않을 때
class InvalidPathError(Exception):
    pass


# error: 확장자가 올바르지 않을 때
class InvalidExtensionError(Exception):
    pass


# error: fps가 올바르지 않을 때
class InvalidFPSError(Exception):
    pass