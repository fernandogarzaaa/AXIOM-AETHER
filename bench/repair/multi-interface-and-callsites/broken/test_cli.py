from helpers import total
from report import summary
from cli import main

assert total([1, 2, 3]) == 6, "expected 6 got %r" % total([1, 2, 3])
assert total([]) == 0
assert summary([1, 2, 3]) == 6
assert summary([]) == 0
main()
