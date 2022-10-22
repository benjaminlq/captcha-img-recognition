from dev import utils

def test_collapse():
    sample = [0, 0, 1, 2, 1, 1, 0, 1, 1, 5, 0, 4, 4]
    assert utils.collapse(sample) == [1, 2, 1, 1, 5, 4], "Incorrect Character Collapse"
    
def test_greedy_decode():
    return