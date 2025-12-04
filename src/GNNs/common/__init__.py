#This makes the common folder immportable (a namespace package)
#   Reviewed the difference between a package and a namespace package today!
#       regular package: has __init__.py, and file is implicity executed and the object is defined to bound names in package
#           I knew calling __init__ as a function initialized an object, but didn't realize it also bound names in the package namespace
#       namespace package: no __init__.py, and files are not executed until explicitly imported
#optional re-exports
from .data_sources import DataSource, LocalJSON
