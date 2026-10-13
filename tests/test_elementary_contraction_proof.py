"""Direct finite GF(2) checks of the cancellation identities used in README."""
from persistence_pairing_graph._lift import boundary_old_chain


def toggle_sum(*supports):
    result = set()
    for support in supports:
        result.symmetric_difference_update(support)
    return result


def test_elementary_filtered_contraction_on_one_square():
    # One closed unit square: doubled grid 3x3, old cell IDs in row-major order.
    # Cancel a=vertex(0,0) against b=edge(0,1).  d(b)=a+vertex(0,2).
    grid = (3, 3)
    a, b = 0, 1
    r = {2}
    surviving = set(range(9)) - {a, b}

    def d(chain):
        return set(map(int, boundary_old_chain(sorted(chain), grid)))

    def p(chain):
        result = set()
        for c in chain:
            term = set() if c == b else (r if c == a else {c})
            result.symmetric_difference_update(term)
        return result

    def i(chain):
        result = set()
        for c in chain:
            assert c in surviving
            term = {c}
            if a in d({c}):
                term.add(b)
            result.symmetric_difference_update(term)
        return result

    def h(chain):
        return {b} if a in chain else set()

    def d_prime(chain):
        return p(d(i(chain)))

    assert d({b}) == {a, 2}
    for c in surviving:
        assert p(i({c})) == {c}
        assert d(i({c})) == i(d_prime({c}))
        assert not d_prime(d_prime({c}))
    for c in range(9):
        original = {c}
        assert p(d(original)) == d_prime(p(original))
        assert toggle_sum(original, i(p(original))) == toggle_sum(d(h(original)), h(d(original)))
