"""
Clean, modular linear evaluator for Mastergoal game.
Receives weights as parameters for easy experimentation.
"""

class LinearEvaluator:
    """
    Modular linear evaluator that receives weights as parameters.
    Separates feature extraction from weight management.
    """
    
    def __init__(self, level, weights=None, use_experiment_weights=False, debug=False):
        """
        Initialize the evaluator with custom or default weights.
        
        Args:
            level: Game level (1, 2, or 3)
            weights: Custom weights dictionary. If provided, takes precedence over other options.
            use_experiment_weights: If True, uses experiment weights; if False, uses default weights.
                                   Only used if weights is None.
            debug: Enable debug output
        """
        self.level = level
        self.debug = debug
        
        # Priority: explicit weights > experiment weights > default weights
        if weights is not None:
            # Use explicitly provided weights (for optimizer)
            self.weights = weights.copy()
        elif use_experiment_weights:
            # Use experiment weights (for launcher)
            from weights_config import EXPERIMENT_WEIGHTS
            self.weights = EXPERIMENT_WEIGHTS[level].copy()
        else:
            # Use default weights (fallback)
            from weights_config import DEFAULT_WEIGHTS
            self.weights = DEFAULT_WEIGHTS[level].copy()
        
        # Validate that we have weights for this level
        if not self.weights:
            raise ValueError(f"No weights provided for level {level}")
    
    def update_weights(self, new_weights):
        """
        Update the weights dictionary.
        
        Args:
            new_weights: Dictionary with new weights
        """
        self.weights.update(new_weights)
    
    def set_weights(self, weights):
        """
        Replace all weights with new ones.
        
        Args:
            weights: New weights dictionary
        """
        self.weights = weights.copy()
    
    def extract_features(self, game, team):
        """
        Extract features based on the game level.
        
        Args:
            game: Game state
            team: Team to evaluate
            
        Returns:
            Dictionary with extracted features
        """
        if self.level == 1:
            return self._extract_features_level1(game, team)
        elif self.level == 2:
            return self._extract_features_level2(game, team)
        else:  # level 3
            return self._extract_features_level3(game, team)
    
    def _extract_features_level1(self, game, team):
        """Extract features for level 1."""
        features = {}
        opponent_team = game.RIGHT if team == game.LEFT else game.LEFT
        mid_row = 7
        
        player = game.get_team_players(team)[0]
        opponent = game.get_team_players(opponent_team)[0]
        ball_pos = game.ball.position
        
        if self.debug:
            print(f"\n[DEBUG - Level 1 - Team {team}]")
            print(f"Ball: {ball_pos}, Player: {player.position}, Opponent: {opponent.position}")
        
        # Ball advancement
        if team == game.LEFT:
            features['avance_balon'] = ball_pos.row - mid_row
        else:
            features['avance_balon'] = mid_row - ball_pos.row
            
        # Distances and positions
        features['dist_balon_jug'] = player.position.distance(ball_pos) - 1
        features['dist_balon_op'] = opponent.position.distance(ball_pos) - 1
        
        # Player and opponent advancement
        features['avance_jug'] = player.position.row - mid_row
        features['avance_op'] = mid_row - opponent.position.row # CHANGED
        
        if team == game.RIGHT:
            features['avance_jug'] *= -1
            features['avance_op'] *= -1
        
        # Column distances
        features['dist_col_balon_jug'] = abs(player.position.col - ball_pos.col)
        features['dist_col_balon_op'] = abs(opponent.position.col - ball_pos.col)
        
        # Who is closer to ball
        features['mas_cerca_balon'] = features['dist_balon_op'] - features['dist_balon_jug']
        
        return features
    
    def _extract_features_level2(self, game, team):
        """Extract features for level 2."""
        features = {}
        opponent_team = game.RIGHT if team == game.LEFT else game.LEFT
        mid_row = 7
        
        players = game.get_team_players(team)
        opponents = game.get_team_players(opponent_team)
        ball_pos = game.ball.position
        
        if self.debug:
            print(f"\n[DEBUG - Level 2 - Team {team}]")
            print(f"Ball: {ball_pos}, Players: {[p.position for p in players]}")
        
        # Ball advancement
        if team == game.LEFT:
            features['avance_balon'] = ball_pos.row - mid_row
        else:
            features['avance_balon'] = mid_row - ball_pos.row
        
        # Distance features
        distances_to_ball = [p.position.distance(ball_pos) - 1 for p in players]
        op_distances_to_ball = [p.position.distance(ball_pos) - 1 for p in opponents]
        
        features['min_dist_balon_jug'] = min(distances_to_ball)
        features['max_dist_balon_jug'] = max(distances_to_ball)
        features['min_dist_balon_op'] = min(op_distances_to_ball)
        features['max_dist_balon_op'] = max(op_distances_to_ball)
        
        # Advancement features
        player_advancements = [p.position.row - mid_row for p in players]
        op_advancements = [mid_row - p.position.row  for p in opponents]
        
        if team == game.RIGHT:
            player_advancements = [-x for x in player_advancements]
            op_advancements = [-x for x in op_advancements]
        
        features['max_avance_jug'] = max(player_advancements)
        features['min_avance_jug'] = min(player_advancements)
        features['max_avance_op'] = max(op_advancements)
        features['min_avance_op'] = min(op_advancements)
        
        # Border proximity
        features['muy_cerca_borde_jug'] = self._calculate_border_proximity(players)
        features['muy_cerca_borde_op'] = self._calculate_border_proximity(opponents)
        
        # Ball control
        features['mas_cerca_balon'] = features['min_dist_balon_op'] - features['min_dist_balon_jug']
        
        # Special tiles
        features['casilla_neutra'] = 1 if game.is_neutral_tile(ball_pos) else 0
        
        # Team formation features
        features['muy_cerca_jug'] = 1 if self._has_close_players(players, 3) else 0
        features['muy_lejos_jug'] = 1 if self._has_distant_players(players, 6) else 0
        features['muy_cerca_op'] = 1 if self._has_close_players(opponents, 3) else 0
        features['muy_lejos_op'] = 1 if self._has_distant_players(opponents, 6) else 0
        
        return features
    
    def _extract_features_level3(self, game, team):
        """Extract features for level 3."""
        features = {}
        opponent_team = game.RIGHT if team == game.LEFT else game.LEFT
        mid_row = 7
        
        players = game.get_team_players(team)
        opponents = game.get_team_players(opponent_team)
        ball_pos = game.ball.position
        
        if self.debug:
            print(f"\n[DEBUG - Level 3 - Team {team}]")
            print(f"Ball: {ball_pos}, Players: {[p.position for p in players]}")
        
        # Ball advancement
        if team == game.LEFT:
            features['avance_balon'] = ball_pos.row - mid_row
        else:
            features['avance_balon'] = mid_row - ball_pos.row
        
        # Sorted distances to ball
        distances_to_ball = sorted([p.position.distance(ball_pos) - 1 for p in players])
        op_distances_to_ball = sorted([p.position.distance(ball_pos) - 1 for p in opponents])
        
        features['dist_balon_jug_1'] = distances_to_ball[0] if len(distances_to_ball) > 0 else 0
        features['dist_balon_jug_2'] = distances_to_ball[1] if len(distances_to_ball) > 1 else 0
        features['dist_balon_jug_3'] = distances_to_ball[2] if len(distances_to_ball) > 2 else 0
        
        features['dist_balon_op_1'] = op_distances_to_ball[0] if len(op_distances_to_ball) > 0 else 0
        features['dist_balon_op_2'] = op_distances_to_ball[1] if len(op_distances_to_ball) > 1 else 0
        features['dist_balon_op_3'] = op_distances_to_ball[2] if len(op_distances_to_ball) > 2 else 0
        
        # Sorted advancements
        player_advancements = [p.position.row - mid_row for p in players]
        op_advancements = [mid_row - p.position.row for p in opponents]
        
        if team == game.RIGHT:
            player_advancements = [-x for x in player_advancements]
            op_advancements = [-x for x in op_advancements]
        
        player_advancements = sorted(player_advancements, reverse=True)
        op_advancements = sorted(op_advancements, reverse=True)
        
        features['avance_jug_1'] = player_advancements[0] if len(player_advancements) > 0 else 0
        features['avance_jug_2'] = player_advancements[1] if len(player_advancements) > 1 else 0
        features['avance_jug_3'] = player_advancements[2] if len(player_advancements) > 2 else 0
        
        features['avance_op_1'] = op_advancements[0] if len(op_advancements) > 0 else 0
        features['avance_op_2'] = op_advancements[1] if len(op_advancements) > 1 else 0
        features['avance_op_3'] = op_advancements[2] if len(op_advancements) > 2 else 0
        
        # Border and area features
        features['muy_cerca_borde_jug'] = self._calculate_border_proximity(players)
        features['muy_cerca_borde_op'] = self._calculate_border_proximity(opponents)
        
        # Goalkeeper features
        goalkeeper = game.get_goalkeeper(team)
        op_goalkeeper = game.get_goalkeeper(opponent_team)
        features['arq_en_area_jug'] = 1 if goalkeeper and game.is_in_small_area(goalkeeper.position, team) else 0
        features['arq_en_area_op'] = 1 if op_goalkeeper and game.is_in_small_area(op_goalkeeper.position, opponent_team) else 0
        
        # Ball control and special tiles
        features['mas_cerca_balon'] = features['dist_balon_op_1'] - features['dist_balon_jug_1']
        features['casilla_neutra'] = 1 if game.is_neutral_tile(ball_pos) else 0
        features['casilla_especial'] = 1 if game.is_special_tile(ball_pos, team) else 0
        
        # Team formation features (count instead of binary)
        features['muy_cerca_jug'] = self._count_close_players(players, 3)
        features['muy_lejos_jug'] = self._count_distant_players(players, 6)
        features['muy_cerca_op'] = self._count_close_players(opponents, 3)
        features['muy_lejos_op'] = self._count_distant_players(opponents, 6)
        
        return features
    
    def _calculate_border_proximity(self, players):
        """Calculate border proximity score for a list of players."""
        score = 0
        for player in players:
            col = player.position.col
            if col == 0 or col == 10:
                score += 2
            elif col == 1 or col == 9:
                score += 1
        return score
    
    def _has_close_players(self, players, threshold):
        """Check if any two players are closer than threshold."""
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                distance = players[i].position.distance(players[j].position) - 1 #LE ESTO LAS CASILLAS DONDE ESTAN
                #print(f"Distance between Player {i+1} and Player {j}: {distance} (Threshold: {int(threshold)})")
                if  distance <= threshold:
                    return True
        return False
    
    def _has_distant_players(self, players, threshold):
        """Check if any two players are farther than threshold."""
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                distance = players[i].position.distance(players[j].position)
                #print(f"Distance between Player {i+1} and Player {j}: {distance} (Threshold: {int(threshold)})")
                if distance >= threshold:
                    return True
        return False
    
    def _count_close_players(self, players, threshold):
        """Count pairs of players closer than threshold."""
        count = 0
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                distance = players[i].position.distance(players[j].position)
                #print(f"Distance between Player {i+1} and Player {j+1}: {distance} (Threshold: {int(threshold)})")
                if distance <= threshold:
                    #print("PLUS ONE CLOSE")
                    count += 1
        return count
    
    def _count_distant_players(self, players, threshold):
        """Count pairs of players farther than threshold."""
        count = 0
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                distance = players[i].position.distance(players[j].position)
                #print(f"Distance between Player {i+1} and Player {j+1}: {distance} (Threshold: {int(threshold)})")
                if distance >= threshold:
                    #print("PLUS ONE")
                    count += 1
        return count
    
    def _calculate_weighted_score(self, features):
        """Calculate weighted score from features."""
        score = 0
        contributions = {}
        
        for feature, value in features.items():
            if feature in self.weights:
                contribution = value * self.weights[feature]
                score += contribution
                contributions[feature] = contribution
                
                if self.debug:
                    print(f"Feature: '{feature}', Value: {value}, Weight: {self.weights[feature]}, Contribution: {contribution}")
        
        return score, contributions
    
    def evaluate(self, game, debug_move=None):
        """
        Evaluate the game state.
        
        Args:
            game: Game state
            debug_move: Move being evaluated (for debug)
            
        Returns:
            Evaluation score (positive favors LEFT, negative favors RIGHT)
        """
        team = game.current_team
        
        if self.debug and debug_move:
            print(f"\n{'='*50}")
            print(f"EVALUATING MOVE: {debug_move}")
            print(f"Current team: {team}")
            print(f"Goals - LEFT: {game.left_goals}, RIGHT: {game.right_goals}")
            print(f"{'='*50}")
        
        # Extract features for both teams
        left_features = self.extract_features(game, game.LEFT)
        right_features = self.extract_features(game, game.RIGHT)
        
        # Calculate scores
        left_score, left_contributions = self._calculate_weighted_score(left_features)
        right_score, right_contributions = self._calculate_weighted_score(right_features)
        
        if self.debug:
            print(f"\n[DEBUG - Final Evaluation]")
            print(f"LEFT score: {left_score}")
            print(f"RIGHT score: {right_score}")
        
        # Final score (LEFT perspective)
        score = left_score - right_score
        
        if self.debug:
            print(f"Raw score (LEFT - RIGHT): {score}")
            print(f"Final score (for {team}): {score if team == game.LEFT else -score}")
        
        # Return from current team's perspective
        return score if team == game.LEFT else -score