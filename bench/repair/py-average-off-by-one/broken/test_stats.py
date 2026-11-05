from stats import average

assert average([2, 4, 6]) == 4.0, "expected 4.0 got %r" % average([2, 4, 6])
assert average([5]) == 5.0
assert average([]) == 0.0
