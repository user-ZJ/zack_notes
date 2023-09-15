import pytest


testcases = [
    ('111', 'aaa'),
    ('222', 'bbb'),
    ('333', 'ccc'),
    ('444', 'ddd'),
    ('555', 'eee'),
]

@pytest.mark.parametrize('test_input, expected', testcases)
def test_data_set(test_input, expected):
    print(test_input,expected)
    assert 1


