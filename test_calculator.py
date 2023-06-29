from calculator import sum, sub, mult, div


# test_soma
def test_sum():
    assert sum(2, 2) == 4


# test_sub
def test_sub():
    assert sub(2, 2) == 0


# test_mult
def test_mult():
    assert mult(2, 2) == 4


# test_div
def test_div():
    assert div(2, 2) == 1
