from main import *
### Testing file for main.py
make_array_tuple = lambda x: tuple(map(tuple, x))
format_array = lambda x: "\n".join(map(lambda y: str(y), x))
### Tests for get_all_plays
# partition: 
# n = 0, n = len(S), 0 < n < len(S)
# i = 0, i = len(S[n]), 0 < i < len(S[n])
# len(S) = 1, len(S) = 2, len(S) > 2
# len(S[n]) = 1, len(S[n]) = 2, len(S[n]) > 2
# each S[n] has the same size, S[n] has different size
###
def test_get_all_plays():
    # n,i = 0, len(S) = 2, len(S[n]) = 2, same size
    S = [[0, 1], [0, 1]]
    all_plays = get_all_plays(S, 0, 0)
    expected = [[[1, 0], [1, 0]], [[1, 0], [0, 1]]]
    if set(map(make_array_tuple, all_plays)) != set(map(make_array_tuple, expected)):
        return False, "Error with 2 players, 2 plays each, n,i = 0\nexpected: " + str(expected) + "\nactual: " + str(all_plays)
    # n,i = max, len(S) > 2, len(S[n]) = 1, >2, variable
    S = [[0], [0, 1, 2], [0, 1, 2, 3], [0, 1]]
    all_plays = get_all_plays(S, 3, 1)
    expected = [[[1],[1,0,0],[1,0,0,0],[0,1]],
                [[1],[1,0,0],[0,1,0,0],[0,1]],
                [[1],[1,0,0],[0,0,1,0],[0,1]],
                [[1],[1,0,0],[0,0,0,1],[0,1]],
                [[1],[0,1,0],[1,0,0,0],[0,1]],
                [[1],[0,1,0],[0,1,0,0],[0,1]],
                [[1],[0,1,0],[0,0,1,0],[0,1]],
                [[1],[0,1,0],[0,0,0,1],[0,1]],
                [[1],[0,0,1],[1,0,0,0],[0,1]],
                [[1],[0,0,1],[0,1,0,0],[0,1]],
                [[1],[0,0,1],[0,0,1,0],[0,1]],
                [[1],[0,0,1],[0,0,0,1],[0,1]]
            ]
    if set(map(make_array_tuple, all_plays)) != set(map(make_array_tuple, expected)):
        return False, "Error with 4 players, 1-4 variable plays each, n,i = (3,1)\nexpected: " + format_array(expected) + "\nactual: " + format_array(all_plays)
    return True, None

def run_all_tests():
    [all_plays, error] = test_get_all_plays()
    if not all_plays:
        print("Error with get all plays: ", error)
    else:
        print("get all plays tests passed")

if __name__ == "__main__":
    run_all_tests()