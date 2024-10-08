from ce_basic import Correlated_equilibrium as ce_basic
from ce_fast import Correlated_equilibrium as ce_fast
from typing import Dict, Callable

def test_strategy_enumeration(Correlated_equilibrium,debug: bool = False):
    ce = Correlated_equilibrium(["a", "b", "c"])
    ce.add_player("1", lambda x: x)
    ce.add_player("2", lambda x: x)
    ce.add_player("3", lambda x: x)

    strategies = ce.get_all_strategy_profiles()
    if debug:
        print(strategies)
    expected_strategies = [{"1": "a", "2": "a", "3": "a"}, {"1": "a", "2": "a", "3": "b"}, {"1": "a", "2": "a", "3": "c"}, 
                           {"1": "a", "2": "b", "3": "a"}, {"1": "a", "2": "b", "3": "b"}, {"1": "a", "2": "b", "3": "c"}, 
                           {"1": "a", "2": "c", "3": "a"}, {"1": "a", "2": "c", "3": "b"}, {"1": "a", "2": "c", "3": "c"}, 
                           {"1": "b", "2": "a", "3": "a"}, {"1": "b", "2": "a", "3": "b"}, {"1": "b", "2": "a", "3": "c"}, 
                           {"1": "b", "2": "b", "3": "a"}, {"1": "b", "2": "b", "3": "b"}, {"1": "b", "2": "b", "3": "c"}, 
                           {"1": "b", "2": "c", "3": "a"}, {"1": "b", "2": "c", "3": "b"}, {"1": "b", "2": "c", "3": "c"}, 
                           {"1": "c", "2": "a", "3": "a"}, {"1": "c", "2": "a", "3": "b"}, {"1": "c", "2": "a", "3": "c"}, 
                           {"1": "c", "2": "b", "3": "a"}, {"1": "c", "2": "b", "3": "b"}, {"1": "c", "2": "b", "3": "c"}, 
                           {"1": "c", "2": "c", "3": "a"}, {"1": "c", "2": "c", "3": "b"}, {"1": "c", "2": "c", "3": "c"}]
    
    assert(strategies == expected_strategies), "Strategy enumeration failed"
    print("Strategy enumeration passed\n")

def dominant_strategy_example(Correlated_equilibrium, debug: bool = False): 
    def get_player_utility(player: str) -> Callable[[Dict[str, str]], float]:
        other_player = "P1" if player == "P2" else "P1"
        strategy_mapping = {"L": 0, "R": 1}
        u = [[3, 1], [5, 7]] if player == "P1" else [[3, 5], [6, 8]]
        def player_utility(profile: Dict[str, str]) -> float:
            return u[strategy_mapping[profile["P1"]]][strategy_mapping[profile["P2"]]]
        return player_utility

    ce = Correlated_equilibrium(["L", "R"], debug)
    ce.add_player("P1", get_player_utility("P1"))
    ce.add_player("P2", get_player_utility("P2"))
    ce.initialize_distribution()
    distribution = ce.optimize_distribution()

    if debug:
        print(distribution)

    for strategy in distribution:
        if(strategy["strategy"] == {"P1": "R", "P2": "R"}):
            assert(strategy["probability"] == 1), "P1 and P2 both playing R should have probability 1"
        else:
            assert(strategy["probability"] == 0), "P1 and P2 both not playing R should have probability 0"
    print("Prof Bryce dominant strategy example passed\n")

def prof_bryce_example(Correlated_equilibrium, debug: bool = False): 
    def get_player_utility(player: str) -> Callable[[Dict[str, str]], float]:
        other_player = "P1" if player == "P2" else "P1"
        strategy_mapping = {"L": 0, "R": 1}
        u = [[3, 1], [2, 7]] if player == "P1" else [[4, 8], [6, 5]]
        def player_utility(profile: Dict[str, str]) -> float:
            return u[strategy_mapping[profile["P1"]]][strategy_mapping[profile["P2"]]]
        return player_utility

    ce = Correlated_equilibrium(["L", "R"], debug)
    ce.add_player("P1", get_player_utility("P1"))
    ce.add_player("P2", get_player_utility("P2"))
    ce.initialize_distribution()
    distribution = ce.optimize_distribution()

    if debug:
        print(distribution)

    for strategy in distribution:
        if(strategy["strategy"] == {"P1": "L", "P2": "L"}):
            assert(abs(strategy["probability"] - 0.171) < 0.01), "P1 and P2 both playing L should have probability 0.171"
        elif strategy["strategy"] == {"P1": "L", "P2": "R"}:
            assert(abs(strategy["probability"] - 0.029) < 0.01), "P1 playing L and P2 playing R should have probability 0.029"
        elif strategy["strategy"] == {"P1": "R", "P2": "L"}:
            assert(abs(strategy["probability"] - 0.686) < 0.01), "P1 playing R and P2 playing L should have probability 0.686"
        elif strategy["strategy"] == {"P1": "R", "P2": "R"}:
            assert(abs(strategy["probability"] - 0.114) < 0.01), "P1 and P2 both playing R should have probability 0.114"
        else:
            assert(False), "Invalid strategy"
    print("Prof Bryce dominant strategy example passed\n")

def game_of_chicken_example(Correlated_equilibrium, debug: bool = False):
    def get_player_utility(player: str) -> Callable[[Dict[str, str]], float]:
        other_player = "P1" if player == "P2" else "P1"
        strategy_mapping = {"D": 0, "C": 1}
        u = [[0, 7], [2, 6]] if player == "P1" else [[0, 2], [7, 6]]
        def player_utility(profile: Dict[str, str]) -> float:
            return u[strategy_mapping[profile["P1"]]][strategy_mapping[profile["P2"]]]
        return player_utility

    ce = Correlated_equilibrium(["D", "C"], debug)
    ce.add_player("P1", get_player_utility("P1"))
    ce.add_player("P2", get_player_utility("P2"))
    ce.initialize_distribution()
    distribution = ce.optimize_distribution()

    if debug:
        print(distribution)

    for strategy in distribution:
        if(strategy["strategy"] == {"P1": "D", "P2": "D"}):
            assert(abs(strategy["probability"] - 0) < 0.01), "P1 and P2 both playing D should have probability 0.171"
        elif strategy["strategy"] == {"P1": "D", "P2": "C"}:
            assert(abs(strategy["probability"] - 0.25) < 0.01), "P1 playing D and P2 playing C should have probability 0.029"
        elif strategy["strategy"] == {"P1": "C", "P2": "D"}:
            assert(abs(strategy["probability"] - 0.25) < 0.01), "P1 playing C and P2 playing D should have probability 0.686"
        elif strategy["strategy"] == {"P1": "C", "P2": "C"}:
            assert(abs(strategy["probability"] - 0.5) < 0.01), "P1 and P2 both playing C should have probability 0.114"
        else:
            assert(False), "Invalid strategy"
    print("Game of chicken example passed\n")

def three_player_game_with_dominant_strategy(Correlated_equilibrium, debug: bool = False):
    def get_player_utility(player: str) -> Callable[[Dict[str, str]], float]:
        all_players = ["P1", "P2", "P3"]
        other_players = [p for p in all_players if p != player]

        u_1 = [[[4, 1, 2], [2, 3, 1], [1, 2, 3]], [[2, 2, 1], [1, 3, 2], [2, 1, 3]], [[1, 1, 3], [1, 2, 3], [3, 1, 2]]]
        u_2 = [[[4, 1, 2], [2, 3, 1], [1, 2, 3]], [[3, 2, 1], [1, 3, 2], [2, 1, 3]], [[2, 1, 3], [1, 2, 3], [3, 1, 2]]]
        u_3 = [[[4, 1, 2], [2, 3, 1], [1, 2, 3]], [[3, 2, 1], [1, 3, 2], [2, 1, 3]], [[2, 1, 3], [1, 2, 3], [3, 1, 2]]]

        strategy_mapping = {"a": 0, "b": 1, "c": 2}
        u = u_1 if player == "P1" else u_2 if player == "P2" else u_3
        def player_utility(profile: Dict[str, str]) -> float:
            def s_to_i(s: str) -> int:
                return strategy_mapping[profile[s]]
            return u[s_to_i("P1")][s_to_i("P2")][s_to_i("P3")]
        return player_utility

    ce = Correlated_equilibrium(["a", "b", "c"], debug)
    for player in ["P1", "P2", "P3"]:
        ce.add_player(player, get_player_utility(player))
    ce.initialize_distribution()
    distribution = ce.optimize_distribution()

    if debug:
        print(distribution)

    for strategy in distribution:
        if(strategy["strategy"] == {"P1": "a", "P2": "a", "P3": "a"}):
            assert(strategy["probability"] == 1), "P1, P2, and P3 all playing a should have probability 1"
        else:
            assert(strategy["probability"] == 0), "P1, P2, and P3 all not playing a should have probability 0"
    print("Three player game with dominant strategy passed\n")

def three_player_game_with_mixed_equilibria(Correlated_equilibrium, debug: bool = False):
    def get_player_utility(player: str) -> Callable[[Dict[str, str]], float]:
        all_players = ["P1", "P2", "P3"]
        other_players = [p for p in all_players if p != player]

        u_1 = [[[1/2, -1, 1], [-1, -3.25, .5], [1, .5, 3.25]], [[3.25, 1, .5], [1, .5, -1], [.5, -1, -3.25]], [[-3.25, .5, -1], [.5, 3.25, 1], [-1, 1, .5]]]
        u_2 = [[[1/2, -1, 1], [3.25, 1, .5], [-3.25, .5, -1]], [[-1, -3.25, .5], [1, .5, -1], [.5, 3.25, 1]], [[1, .5, 3.25], [.5, -1, -3.25], [-1, 1, .5]]]
        u_3 = [[[1/2, 3.25, -3.25], [-1, 1, .5], [1, .5, -1]], [[-1, 1, .5], [-3.25, .5, 3.25], [.5, -1, 1]], [[1, .5, -1], [.5, -1, 1], [3.25, -3.25, .5]]]

        strategy_mapping = {"r": 0, "p": 1, "s": 2}
        u = u_1 if player == "P1" else u_2 if player == "P2" else u_3
        def player_utility(profile: Dict[str, str]) -> float:
            def s_to_i(s: str) -> int:
                return strategy_mapping[profile[s]]
            return u[s_to_i("P1")][s_to_i("P2")][s_to_i("P3")]
        return player_utility

    ce = Correlated_equilibrium(["r", "p", "s"], debug)
    for player in ["P1", "P2", "P3"]:
        ce.add_player(player, get_player_utility(player))
    ce.initialize_distribution()
    distribution = ce.optimize_distribution()

    if debug:
        print(distribution)

    expected = [{'probability': 0.05042016806722689, 'strategy': {'P1': 'r', 'P2': 'r', 'P3': 'r'}}, {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'r', 'P3': 'p'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'r', 'P3': 's'}}, {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'p', 'P3': 'r'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'p', 'P3': 'p'}}, {'probability': 0.37815126050420167, 'strategy': {'P1': 'r', 'P2': 'p', 'P3': 's'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 's', 'P3': 'r'}}, {'probability': -0.0, 'strategy': {'P1': 'r', 'P2': 's', 'P3': 'p'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 's', 'P3': 's'}}, {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'r', 'P3': 'r'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'r', 'P3': 'p'}}, {'probability': -0.0, 'strategy': {'P1': 'p', 'P2': 'r', 'P3': 's'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'p', 'P3': 'r'}}, {'probability': 0.0504201680672269, 'strategy': {'P1': 'p', 'P2': 'p', 'P3': 'p'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'p', 'P3': 's'}}, {'probability': 0.09243697478991597, 'strategy': {'P1': 'p', 'P2': 's', 'P3': 'r'}}, 
                {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 's', 'P3': 'p'}}, {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 's', 'P3': 's'}}, 
                {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'r', 'P3': 'r'}}, {'probability': 0.3781512605042017, 'strategy': {'P1': 's', 'P2': 'r', 'P3': 'p'}}, 
                {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'r', 'P3': 's'}}, {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'p', 'P3': 'r'}}, 
                {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'p', 'P3': 'p'}}, {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'p', 'P3': 's'}}, 
                {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 's', 'P3': 'r'}}, {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 's', 'P3': 'p'}}, 
                {'probability': 0.0504201680672269, 'strategy': {'P1': 's', 'P2': 's', 'P3': 's'}}]
    for i in distribution:
        for j in expected:
            if i["strategy"] == j["strategy"]:
                assert(abs(i["probability"] - j["probability"]) < 0.01), "Probabilities do not match for strategy " + str(i["strategy"])
    print("Three player game with mixed equilibria passed\n")

def three_player_game_with_dominant_equilibria_2(Correlated_equilibrium, debug: bool = False):
    def get_player_utility(player: str) -> Callable[[Dict[str, str]], float]:
        all_players = ["P1", "P2", "P3"]
        other_players = [p for p in all_players if p != player]

        u_1 = [[[5,2,1],[3,2,1],[4,3,2]],[[3,2,1],[4,6,5],[3,2,1]],[[5,4,3],[4,3,2],[5,4,7]]]
        u_2 = [[[5,2,1],[4,3,2],[1,1,1]],[[5,3,1],[4,6,5],[1,1,1]],[[5,4,3],[3,3,3],[1,1,7]]]
        u_3 = [[[5,3,6],[5,3,6],[5,3,6]],[[5,3,6],[5,6,6],[5,3,6]],[[5,3,6],[5,3,6],[5,3,7]]]

        strategy_mapping = {"a": 0, "b": 1, "c": 2}
        u = u_1 if player == "P1" else u_2 if player == "P2" else u_3
        def player_utility(profile: Dict[str, str]) -> float:
            def s_to_i(s: str) -> int:
                return strategy_mapping[profile[s]]
            return u[s_to_i("P1")][s_to_i("P2")][s_to_i("P3")]
        return player_utility

    ce = Correlated_equilibrium(["a", "b", "c"], debug)
    for player in ["P1", "P2", "P3"]:
        ce.add_player(player, get_player_utility(player))
    ce.initialize_distribution()
    distribution = ce.optimize_distribution()

    if debug:
        print(distribution)

    # expected = [{'probability': 0.05042016806722689, 'strategy': {'P1': 'r', 'P2': 'r', 'P3': 'r'}}, {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'r', 'P3': 'p'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'r', 'P3': 's'}}, {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'p', 'P3': 'r'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 'p', 'P3': 'p'}}, {'probability': 0.37815126050420167, 'strategy': {'P1': 'r', 'P2': 'p', 'P3': 's'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 's', 'P3': 'r'}}, {'probability': -0.0, 'strategy': {'P1': 'r', 'P2': 's', 'P3': 'p'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'r', 'P2': 's', 'P3': 's'}}, {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'r', 'P3': 'r'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'r', 'P3': 'p'}}, {'probability': -0.0, 'strategy': {'P1': 'p', 'P2': 'r', 'P3': 's'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'p', 'P3': 'r'}}, {'probability': 0.0504201680672269, 'strategy': {'P1': 'p', 'P2': 'p', 'P3': 'p'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 'p', 'P3': 's'}}, {'probability': 0.09243697478991597, 'strategy': {'P1': 'p', 'P2': 's', 'P3': 'r'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 's', 'P3': 'p'}}, {'probability': 0.0, 'strategy': {'P1': 'p', 'P2': 's', 'P3': 's'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'r', 'P3': 'r'}}, {'probability': 0.3781512605042017, 'strategy': {'P1': 's', 'P2': 'r', 'P3': 'p'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'r', 'P3': 's'}}, {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'p', 'P3': 'r'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'p', 'P3': 'p'}}, {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 'p', 'P3': 's'}}, 
    #             {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 's', 'P3': 'r'}}, {'probability': 0.0, 'strategy': {'P1': 's', 'P2': 's', 'P3': 'p'}}, 
    #             {'probability': 0.0504201680672269, 'strategy': {'P1': 's', 'P2': 's', 'P3': 's'}}]
    # for i in distribution:
    #     for j in expected:
    #         if i["strategy"] == j["strategy"]:
    #             assert(abs(i["probability"] - j["probability"]) < 0.01), "Probabilities do not match for strategy " + str(i["strategy"])
    print("Three player game with dominant strategy passed\n")



    

if __name__ == "__main__":
    print("RUNNING CORRELATED EQUILIBRIUM TESTS...\n\n")
    for ce in [ce_basic, ce_fast]:
        # enumerates all possible strategy combinations for 3 players, 3 strategies
        print("Testing strategy enumeration...")
        test_strategy_enumeration(ce)

        """
        TESTING EXAMPLES FOR 2 PLAYERS, 2 STRATEGIES
        """
        # dominant strategy example where P(R,R) = 1
        print("Testing dominant strategy example...")
        dominant_strategy_example(ce)

        # prof bryce example from youtube video
        print("Testing prof bryce example...")
        prof_bryce_example(ce)

        # game of chicken example from wikipedia (symmetric)
        print("Testing game of chicken example...")
        game_of_chicken_example(ce)

        """
        TESTING EXAMPLES FOR 3 PLAYERS, 3 STRATEGIES
        """
        # 3 player game where P(a,a,a) = 1
        print("Testing 3 player game with dominant strategy...")
        three_player_game_with_dominant_strategy(ce)

        # 3 playerr game with mixed equilibria
        print("Testing 3 player game with mixed equilibria...")
        three_player_game_with_mixed_equilibria(ce)

        # 3 player game with mixed equilibria 2
        print("Testing 3 player game with dominant equilibria 2...")
        three_player_game_with_dominant_equilibria_2(ce)
