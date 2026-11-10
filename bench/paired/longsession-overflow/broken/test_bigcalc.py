from bigcalc import square, total

assert square(12) == 144
assert square(0) == 0
assert total([1, 2, 3]) == 6, "expected 6 got %r" % total([1, 2, 3])
assert total([]) == 0
