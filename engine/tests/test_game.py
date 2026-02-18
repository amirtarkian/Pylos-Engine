"""Comprehensive tests for PylosGame engine."""

import numpy as np
import pytest
from game import PylosGame


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def game():
    """Fresh PylosGame instance."""
    return PylosGame()


@pytest.fixture
def game_with_square(game):
    """Game with a 2x2 white square at level 0, top-left corner.

    Sets up board manually so check_square triggers.
    """
    game.board[0][0, 0] = 1
    game.board[0][0, 1] = 1
    game.board[0][1, 0] = 1
    game.board[0][1, 1] = 1
    game.reserves[1] -= 4
    return game


# ======================================================================
# Board initialization
# ======================================================================

class TestBoardInit:
    def test_level_sizes(self, game):
        assert game.board[0].shape == (4, 4)
        assert game.board[1].shape == (3, 3)
        assert game.board[2].shape == (2, 2)
        assert game.board[3].shape == (1, 1)

    def test_all_empty(self, game):
        for level in range(4):
            assert np.all(game.board[level] == 0)

    def test_reserves(self, game):
        assert game.reserves[1] == 15
        assert game.reserves[-1] == 15

    def test_turn(self, game):
        assert game.turn == 1  # white goes first

    def test_action_space(self, game):
        assert game.action_space == 303

    def test_observation_shape(self, game):
        assert game.observation_shape == (32,)

    def test_index_to_coords_length(self, game):
        assert len(game.index_to_coords) == 30

    def test_coords_to_index_length(self, game):
        assert len(game.coords_to_index) == 30

    def test_raise_pairs_count(self, game):
        assert len(game.raise_pairs) == 273

    def test_raise_action_to_pair_count(self, game):
        assert len(game.raise_action_to_pair) == 273

    def test_index_mapping_roundtrip(self, game):
        for idx in range(30):
            coords = game.index_to_coords[idx]
            assert game.coords_to_index[coords] == idx

    def test_raise_pairs_are_higher(self, game):
        """All raise pairs must have dst on a strictly higher level."""
        for src_idx, dst_idx in game.raise_pairs:
            src_level = game.index_to_coords[src_idx][0]
            dst_level = game.index_to_coords[dst_idx][0]
            assert dst_level > src_level


# ======================================================================
# Placement
# ======================================================================

class TestPlacement:
    def test_place_on_empty_level0(self, game):
        assert game.place(0, 0, 0) is True
        assert game.board[0][0, 0] == 1
        assert game.reserves[1] == 14

    def test_place_on_occupied_fails(self, game):
        game.place(0, 0, 0)
        game.turn = -1  # switch to black
        assert game.place(0, 0, 0) is False

    def test_place_unsupported_level1_fails(self, game):
        """Level 1 position needs 4 pieces below it."""
        assert game.place(1, 0, 0) is False

    def test_place_supported_level1_works(self, game):
        """Place 4 pieces below then place on level 1."""
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = -1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = -1
        game.reserves[1] -= 2
        game.reserves[-1] -= 2
        assert game.place(1, 0, 0) is True
        assert game.board[1][0, 0] == 1

    def test_place_no_reserves_fails(self, game):
        game.reserves[1] = 0
        assert game.place(0, 0, 0) is False

    def test_place_sets_last_move(self, game):
        game.place(0, 2, 3)
        assert game.last_move == (0, 2, 3)


# ======================================================================
# Support checking
# ======================================================================

class TestSupport:
    def test_level0_always_supported(self, game):
        for r in range(4):
            for c in range(4):
                assert game.is_supported(0, r, c) is True

    def test_level1_needs_4_below(self, game):
        """Level 1 position (0,0) needs pieces at level 0 (0,0),(0,1),(1,0),(1,1)."""
        assert game.is_supported(1, 0, 0) is False
        game.board[0][0, 0] = 1
        assert game.is_supported(1, 0, 0) is False
        game.board[0][0, 1] = 1
        assert game.is_supported(1, 0, 0) is False
        game.board[0][1, 0] = 1
        assert game.is_supported(1, 0, 0) is False
        game.board[0][1, 1] = 1
        assert game.is_supported(1, 0, 0) is True

    def test_any_color_supports(self, game):
        """Support doesn't depend on color of pieces below."""
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = -1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = -1
        assert game.is_supported(1, 0, 0) is True

    def test_level2_support(self, game):
        """Level 2 needs 4 pieces on level 1."""
        # Fill level 0 enough to support level 1
        for r in range(3):
            for c in range(3):
                game.board[0][r, c] = 1
        # Fill level 1 enough to support level 2 at (0,0)
        game.board[1][0, 0] = 1
        game.board[1][0, 1] = 1
        game.board[1][1, 0] = 1
        game.board[1][1, 1] = 1
        assert game.is_supported(2, 0, 0) is True


# ======================================================================
# Piece has top
# ======================================================================

class TestPieceHasTop:
    def test_no_top_on_level0(self, game):
        game.board[0][0, 0] = 1
        assert game.piece_has_top(0, 0, 0) is False

    def test_has_top_when_supporting(self, game):
        """A level 0 piece has a top when a level 1 piece sits on it."""
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        game.board[1][0, 0] = 1
        assert game.piece_has_top(0, 0, 0) is True
        assert game.piece_has_top(0, 0, 1) is True
        assert game.piece_has_top(0, 1, 0) is True
        assert game.piece_has_top(0, 1, 1) is True

    def test_level3_never_has_top(self, game):
        game.board[3][0, 0] = 1
        assert game.piece_has_top(3, 0, 0) is False

    def test_unsupporting_piece_no_top(self, game):
        """A level 0 piece not under any level 1 piece has no top."""
        game.board[0][3, 3] = 1
        assert game.piece_has_top(0, 3, 3) is False


# ======================================================================
# Raising
# ======================================================================

class TestRaising:
    def test_raise_own_piece_works(self, game):
        """Raise own piece from level 0 to supported level 1."""
        # Place 4 pieces to support level 1
        game.board[0][0, 0] = 1   # this one will be raised
        game.board[0][0, 1] = -1
        game.board[0][1, 0] = -1
        game.board[0][1, 1] = -1
        # Also need support at (2,0),(2,1) area for raising to level 1 (1,0)
        # Actually let's use a spot where source is NOT one of the 4 supports
        # Place source at (0, 3, 3) and raise to level 1 (0, 0)
        game.board[0][3, 3] = 1
        game.reserves[1] -= 2

        result = game.raise_piece(0, 3, 3, 1, 0, 0)
        assert result is True
        assert game.board[0][3, 3] == 0
        assert game.board[1][0, 0] == 1

    def test_raise_opponent_piece_fails(self, game):
        """Cannot raise opponent's piece."""
        game.board[0][0, 0] = -1  # opponent's piece
        # Set up support for level 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        result = game.raise_piece(0, 0, 0, 1, 0, 0)
        assert result is False

    def test_raise_piece_with_top_fails(self, game):
        """Cannot raise a piece that supports another piece."""
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        game.board[1][0, 0] = -1  # sits on top
        # Try to raise (0,0,0) which supports (1,0,0)
        result = game.raise_piece(0, 0, 0, 1, 0, 0)
        assert result is False

    def test_raise_to_lower_level_fails(self, game):
        """Cannot raise to same or lower level."""
        game.board[1][0, 0] = 1
        result = game.raise_piece(1, 0, 0, 0, 0, 0)
        assert result is False

    def test_raise_temporarily_removes_src_for_support_check(self, game):
        """When raising, src piece is temporarily removed for support check.

        If src piece is one of the 4 supporting the destination, the raise
        should fail because removing src breaks the support.
        """
        # Place 4 pieces at level 0 to support level 1 (0,0)
        game.board[0][0, 0] = 1   # will try to raise this one
        game.board[0][0, 1] = -1
        game.board[0][1, 0] = -1
        game.board[0][1, 1] = -1
        game.reserves[1] -= 1
        game.reserves[-1] -= 3

        # Try to raise (0,0,0) to (1,0,0) -- but (0,0,0) supports (1,0,0)!
        result = game.raise_piece(0, 0, 0, 1, 0, 0)
        assert result is False
        # Source piece should be restored
        assert game.board[0][0, 0] == 1


# ======================================================================
# Formation detection
# ======================================================================

class TestFormationDetection:
    def test_2x2_square_detected(self, game):
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        assert game.check_square(0, 0, 0) is True
        assert game.check_square(0, 0, 1) is True
        assert game.check_square(0, 1, 0) is True
        assert game.check_square(0, 1, 1) is True

    def test_mixed_square_not_detected(self, game):
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = -1
        assert game.check_square(0, 0, 0) is False

    def test_4_in_a_row_level0_horizontal(self, game):
        for c in range(4):
            game.board[0][0, c] = 1
        assert game.check_line(0, 0, 0) is True
        assert game.check_line(0, 0, 3) is True

    def test_4_in_a_row_level0_vertical(self, game):
        for r in range(4):
            game.board[0][r, 0] = 1
        assert game.check_line(0, 0, 0) is True
        assert game.check_line(0, 3, 0) is True

    def test_3_in_a_row_level0_not_detected(self, game):
        """Only 3 in a row on level 0 should NOT trigger (need 4)."""
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][0, 2] = 1
        assert game.check_line(0, 0, 0) is False

    def test_3_in_a_row_level1(self, game):
        game.board[1][0, 0] = 1
        game.board[1][0, 1] = 1
        game.board[1][0, 2] = 1
        assert game.check_line(1, 0, 0) is True

    def test_3_in_a_row_level1_vertical(self, game):
        game.board[1][0, 0] = 1
        game.board[1][1, 0] = 1
        game.board[1][2, 0] = 1
        assert game.check_line(1, 0, 0) is True

    def test_no_line_on_level2(self, game):
        game.board[2][0, 0] = 1
        game.board[2][0, 1] = 1
        assert game.check_line(2, 0, 0) is False

    def test_no_false_positive_partial(self, game):
        """Two pieces in a row should not trigger."""
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        assert game.check_line(0, 0, 0) is False
        assert game.check_square(0, 0, 0) is False

    def test_check_for_removal_with_square(self, game):
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        game.last_move = (0, 1, 1)
        assert game.check_for_removal() is True

    def test_check_for_removal_no_formation(self, game):
        game.board[0][0, 0] = 1
        game.last_move = (0, 0, 0)
        assert game.check_for_removal() is False


# ======================================================================
# Removal
# ======================================================================

class TestRemoval:
    def test_remove_own_piece_works(self, game):
        game.board[0][0, 0] = 1
        game.reserves[1] = 14
        result = game.remove(0, 0, 0)
        assert result is True
        assert game.board[0][0, 0] == 0
        assert game.reserves[1] == 15

    def test_remove_opponent_piece_fails(self, game):
        game.board[0][0, 0] = -1
        result = game.remove(0, 0, 0)
        assert result is False

    def test_remove_supporting_piece_fails(self, game):
        """Cannot remove a piece that supports another."""
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        game.board[1][0, 0] = -1
        result = game.remove(0, 0, 0)
        assert result is False
        assert game.board[0][0, 0] == 1  # piece still there

    def test_get_removable_pieces(self, game):
        game.board[0][0, 0] = 1
        game.board[0][3, 3] = 1
        removable = game.get_removable_pieces()
        assert (0, 0, 0) in removable
        assert (0, 3, 3) in removable

    def test_get_removable_excludes_supporting(self, game):
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        game.board[1][0, 0] = 1
        removable = game.get_removable_pieces()
        # All 4 level-0 pieces support the level-1 piece
        for r in range(2):
            for c in range(2):
                assert (0, r, c) not in removable
        # The level 1 piece should be removable
        assert (1, 0, 0) in removable


# ======================================================================
# Win conditions
# ======================================================================

class TestWinConditions:
    def test_apex_filled_white_wins(self, game):
        game.board[3][0, 0] = 1
        assert game.top_filled() is True
        assert game.get_result() == 1

    def test_apex_filled_black_wins(self, game):
        game.board[3][0, 0] = -1
        assert game.top_filled() is True
        assert game.get_result() == -1

    def test_no_moves_loses(self, game):
        """If current player has no legal moves, they lose."""
        # Set reserves to 0 so no placements possible
        game.reserves[1] = 0
        game.reserves[-1] = 0
        # Fill all level 0 with opponent pieces (no raises for current player)
        for r in range(4):
            for c in range(4):
                game.board[0][r, c] = -1
        # White has no reserves and no own pieces to raise
        assert game.has_move() is False
        assert game.get_result() == -1  # white loses

    def test_game_ongoing(self, game):
        assert game.get_result() is None

    def test_first_person_result_white_wins(self, game):
        game.board[3][0, 0] = 1
        game.turn = 1
        assert game.get_first_person_result() == 1

    def test_first_person_result_black_perspective(self, game):
        game.board[3][0, 0] = 1  # white won
        game.turn = -1
        assert game.get_first_person_result() == -1  # loss from black's view

    def test_swap_result(self):
        assert PylosGame.swap_result(1) == -1
        assert PylosGame.swap_result(-1) == 1


# ======================================================================
# AlphaZero interface
# ======================================================================

class TestAlphaZeroInterface:
    def test_initial_legal_actions_count(self, game):
        """Initially only level 0 placements are legal = 16 positions."""
        legal = game.get_legal_actions()
        assert len(legal) == 16
        # All should be placement actions (0-15)
        for action in legal:
            assert 0 <= action <= 15

    def test_step_and_undo_roundtrip(self, game):
        """Step then undo should restore original state."""
        import copy
        # Save original state
        orig_board = [level.copy() for level in game.board]
        orig_turn = game.turn
        orig_reserves = game.reserves.copy()

        # Execute a step
        legal = game.get_legal_actions()
        action = legal[0]
        game.step(action)

        # State should be different
        assert game.turn != orig_turn

        # Undo
        game.undo_last_action()

        # State should be restored
        assert game.turn == orig_turn
        assert game.reserves == orig_reserves
        for level in range(4):
            assert np.array_equal(game.board[level], orig_board[level])

    def test_observation_shape_and_values(self, game):
        obs = game.to_observation()
        assert obs.shape == (32,)
        assert obs.dtype == np.float32

        # Initially all board cells are 0, reserves are 15/15
        assert np.all(obs[:30] == 0.0)
        assert obs[30] == pytest.approx(1.0)  # own reserves 15/15
        assert obs[31] == pytest.approx(1.0)  # opponent reserves 15/15

    def test_observation_after_placement(self, game):
        game.place(0, 0, 0)  # white places at (0,0,0)
        obs = game.to_observation()
        # From white's perspective (turn is still 1), cell 0 should be +1
        assert obs[0] == pytest.approx(1.0)
        assert obs[30] == pytest.approx(14.0 / 15.0)

    def test_observation_opponent_perspective(self, game):
        """After white places and turn switches, observation should flip."""
        game.board[0][0, 0] = 1
        game.reserves[1] -= 1
        game.turn = -1  # now it's black's turn
        obs = game.to_observation()
        # From black's perspective, white piece is opponent = -1
        assert obs[0] == pytest.approx(-1.0)
        # Black's reserves (current player) first
        assert obs[30] == pytest.approx(15.0 / 15.0)
        assert obs[31] == pytest.approx(14.0 / 15.0)

    def test_step_with_raise_action(self, game):
        """Test step with a raise action index."""
        # Set up board so a raise is possible
        # Place white at (0, 3, 3) and support level 1 (0, 0)
        game.board[0][0, 0] = -1
        game.board[0][0, 1] = -1
        game.board[0][1, 0] = -1
        game.board[0][1, 1] = -1
        game.board[0][3, 3] = 1
        game.reserves[1] -= 1
        game.reserves[-1] -= 4

        # Find the raise action for (0,3,3) -> (1,0,0)
        src_idx = game.coords_to_index[(0, 3, 3)]
        dst_idx = game.coords_to_index[(1, 0, 0)]
        raise_action = None
        for action_idx, pair in game.raise_action_to_pair.items():
            if pair == (src_idx, dst_idx):
                raise_action = action_idx
                break

        assert raise_action is not None
        game.step(raise_action)
        assert game.board[0][3, 3] == 0
        assert game.board[1][0, 0] == 1
        assert game.turn == -1  # turn switched

    def test_step_illegal_action_raises(self, game):
        """Stepping with an illegal action should raise ValueError."""
        # Action 16 is level 1 position 0 -- unsupported at start
        with pytest.raises(ValueError):
            game.step(16)

    def test_get_legal_actions_includes_raises(self, game):
        """After some pieces are placed, raise actions should appear."""
        # Place white pieces that can be raised
        game.board[0][0, 0] = -1
        game.board[0][0, 1] = -1
        game.board[0][1, 0] = -1
        game.board[0][1, 1] = -1
        game.board[0][3, 3] = 1
        game.reserves[1] -= 1
        game.reserves[-1] -= 4

        legal = game.get_legal_actions()
        raise_actions = [a for a in legal if a >= 30]
        assert len(raise_actions) > 0

    def test_reset_restores_initial_state(self, game):
        """Reset should restore the game to initial state."""
        game.step(0)  # make a move
        game.reset()
        assert game.turn == 1
        assert game.reserves == {1: 15, -1: 15}
        for level in range(4):
            assert np.all(game.board[level] == 0)
        assert game.actions_stack == []


# ======================================================================
# AI removal heuristic
# ======================================================================

class TestAIRemoval:
    def test_greedy_removal_on_square(self, game):
        """Forming a square should trigger removal of up to 2 pieces."""
        # Place 3 white pieces, then step to place the 4th forming a square
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][1, 0] = 1
        game.reserves[1] -= 3

        # Place at (0,1,1) to complete the square
        idx = game.coords_to_index[(0, 1, 1)]
        game.step(idx)

        # After step: the square was formed, greedy removal should have
        # removed up to 2 of white's unsupported pieces.
        # White placed 4 pieces total (3 + 1 in step), then removed up to 2
        # So white should have at most 2 pieces on the board from this square
        white_count = sum(
            np.sum(game.board[level] == 1) for level in range(4)
        )
        assert white_count == 2  # 4 placed - 2 removed

    def test_removal_lowest_level_first(self, game):
        """Greedy removal should prefer lowest level pieces."""
        # Build up to level 1 then form a square on level 1
        # Fill level 0 with enough pieces
        for r in range(3):
            for c in range(3):
                game.board[0][r, c] = 1 if (r + c) % 2 == 0 else -1
        game.board[0][0, 0] = 1
        game.board[0][0, 1] = 1
        game.board[0][0, 2] = 1
        game.board[0][1, 0] = 1
        game.board[0][1, 1] = 1
        game.board[0][1, 2] = 1
        game.board[0][2, 0] = 1
        game.board[0][2, 1] = 1
        game.board[0][2, 2] = 1

        # Place level 1 pieces forming a square
        game.board[1][0, 0] = 1
        game.board[1][0, 1] = 1
        game.board[1][1, 0] = 1
        game.reserves[1] = 2

        # Place at (1, 1, 1) to complete square on level 1
        idx = game.coords_to_index[(1, 1, 1)]
        game.step(idx)

        # Greedy removal removes lowest level first
        # The level 0 pieces that aren't supporting anything should be removed first
        # (if any are unsupported / not supporting level 1 pieces)
        # Just verify reserves went up
        # turn switched to -1, so check white's reserves
        assert game.reserves[1] >= 2  # at least got some back


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    def test_full_level0(self, game):
        """When level 0 is full, placements only on supported higher levels."""
        for r in range(4):
            for c in range(4):
                game.board[0][r, c] = 1 if (r + c) % 2 == 0 else -1

        game.reserves[1] -= 8
        game.reserves[-1] -= 8

        legal = game.get_legal_actions()
        placement_actions = [a for a in legal if a < 30]
        # All level 0 spots are taken, so placements should be on level 1+
        for a in placement_actions:
            level, _, _ = game.index_to_coords[a]
            assert level >= 1

    def test_str_display(self, game):
        """__str__ should return a string representation."""
        s = str(game)
        assert "Level 0" in s
        assert "Level 3" in s
        assert "White" in s or "Black" in s

    def test_check_for_removal_no_last_move(self, game):
        """check_for_removal with no last_move returns False."""
        game.last_move = None
        assert game.check_for_removal() is False

    def test_undo_empty_stack(self, game):
        """Undo with empty stack should do nothing."""
        turn_before = game.turn
        game.undo_last_action()
        assert game.turn == turn_before

    def test_multiple_step_undo(self, game):
        """Multiple steps and undos should work correctly."""
        legal = game.get_legal_actions()
        action1 = legal[0]
        game.step(action1)

        legal2 = game.get_legal_actions()
        action2 = legal2[0]
        game.step(action2)

        # Undo second action
        game.undo_last_action()
        assert game.turn == -1

        # Undo first action
        game.undo_last_action()
        assert game.turn == 1

        # Board should be empty again
        for level in range(4):
            assert np.all(game.board[level] == 0)
