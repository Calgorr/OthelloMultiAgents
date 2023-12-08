from Agents import Agent
import util
import random
from Game import GameState
import sys


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.index = 0  # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(
            self.index
        )


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", **kwargs):
        self.index = 0  # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state: GameState):
        return self.minimax(state, 0, self.depth)[1]

    def max_value(self, state: GameState, agentIndex, depth):
        actions = []
        for action in state.getLegalActions(agentIndex):
            actions.append(
                (
                    self.minimax(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        depth,
                    )[0],
                    action,
                )
            )
        return max(actions)

    def min_value(self, state: GameState, agentIndex, depth):
        actions = []
        for action in state.getLegalActions(agentIndex):
            actions.append(
                (
                    self.minimax(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        depth,
                    )[0],
                    action,
                )
            )
        return min(actions)

    def minimax(self, state: GameState, agentIndex, depth):
        if state.isWin() or depth == 0:
            return (self.evaluationFunction(state), "end")

        agents_num = state.getNumAgents()
        agentIndex %= agents_num
        if agentIndex == agents_num - 1:
            depth -= 1

        if agentIndex == 0:  # Our agent which is the maximizer
            return self.max_value(state, agentIndex, depth)
        else:  # Opponent agent which is the minimizer
            return self.min_value(state, agentIndex, depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            v = self.minimax(
                gameState.generateSuccessor(agentIndex, action),
                agentIndex + 1,
                depth,
                alpha,
                beta,
            )[0]
            actions.append((v, action))
            if v > beta:
                return (v, action)
            alpha = max(alpha, v)
        return max(actions)

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            v = self.minimax(
                gameState.generateSuccessor(agentIndex, action),
                agentIndex + 1,
                depth,
                alpha,
                beta,
            )[0]
            actions.append((v, action))
            if v < alpha:
                return (v, action)
            beta = min(beta, v)
        return min(actions)

    def minimax(self, gameState, agentIndex, depth, alpha=-99999, beta=99999):
        if gameState.isWin() or depth == 0:
            return (self.evaluationFunction(gameState), "end")

        agents_num = gameState.getNumAgents()
        agentIndex %= agents_num
        if agentIndex == agents_num - 1:
            depth -= 1

        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        return self.Expectimax(gameState, 0, self.depth)[1]

    def maxValue(self, gameState, agentIndex, depth):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            actions.append(
                (
                    self.Expectimax(
                        gameState.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        depth,
                    )[0],
                    action,
                )
            )
        return max(actions)

    def minValue(self, gameState, agentIndex, depth):
        actions = []
        total = 0
        for action in gameState.getLegalActions(agentIndex):
            v = self.Expectimax(
                gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth
            )[0]
            total += v
            actions.append((v, action))

        return (total / len(actions),)

    def Expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or depth == 0:
            return (self.evaluationFunction(gameState), "end")

        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1

        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    """

    return (
        0.1 * coin_parity(currentGameState)
        + 0.35 * corners_captured(currentGameState)
        + 0.35 * mobility(currentGameState)
        + 0.2 * stability(currentGameState)
    )


better = betterEvaluationFunction


def coin_parity(current_game_state: GameState):
    result = 0
    result += len(current_game_state.getPieces(0))
    for i in range(1, current_game_state.getNumAgents()):
        result -= len(current_game_state.getPieces(i))
    return 100 * result


def mobility(current_game_state: GameState):
    actual_mobility = 0
    for i in range(current_game_state.getNumAgents()):
        actual_mobility += len(current_game_state.getLegalActions(i))
    if actual_mobility != 0:
        min_players_mobility = actual_mobility - len(
            current_game_state.getLegalActions(0)
        )
        heuristic_mobility = (
            len(current_game_state.getLegalActions(0)) - min_players_mobility
        ) / actual_mobility
        return 100 * heuristic_mobility
    return 0


def corners_captured(current_game_state: GameState):
    max_players_corners = 0
    min_players_corners = 0
    current_game_state.getLegalActions(0)
    for x, y in current_game_state.getLegalActions(0):
        if (
            (x == 0 and y == 0)
            or (x == 0 and y == 7)
            or (x == 7 and y == 0)
            or (x == 7 and y == 7)
        ):
            max_players_corners += 1
    for i in range(1, current_game_state.getNumAgents()):
        for x, y in current_game_state.getLegalActions(i):
            if (
                (x == 0 and y == 0)
                or (x == 0 and y == 7)
                or (x == 7 and y == 0)
                or (x == 7 and y == 7)
            ):
                min_players_corners += 1
    if max_players_corners + min_players_corners != 0:
        return (
            100
            * (max_players_corners - min_players_corners)
            / (max_players_corners + min_players_corners)
        )
    return 0


def stability(current_game_state: GameState):
    corners = current_game_state.getCorners()
    max_stable_pieces = 0
    min_stable_pieces = 0
    for i in corners:
        if i == 0:
            max_stable_pieces += 1
        elif i != -1:
            min_stable_pieces += 1
    max_unstable_pieces = 0
    min_unstbale_pieces = 0
    for pos in current_game_state.getLegalActions(0):
        for i in range(1, current_game_state.getNumAgents()):
            if pos in current_game_state.getLegalActions(i):
                max_unstable_pieces += 1
    for i in range(1, current_game_state.getNumAgents()):
        for pos in current_game_state.getLegalActions(i):
            if pos in current_game_state.getLegalActions(0):
                min_unstbale_pieces += 1
    if (
        max_stable_pieces
        + min_stable_pieces
        + max_unstable_pieces
        + min_unstbale_pieces
        != 0
    ):
        return (
            100
            * (
                (max_stable_pieces - min_unstbale_pieces)
                - (min_stable_pieces - min_unstbale_pieces)
            )
            / (
                max_stable_pieces
                + min_stable_pieces
                + max_unstable_pieces
                + min_unstbale_pieces
            )
        )
    return 0
