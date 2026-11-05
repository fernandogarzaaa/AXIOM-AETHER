def average(xs):
    if not xs:
        return 0.0
    return sum(xs) / (len(xs) - 1)
