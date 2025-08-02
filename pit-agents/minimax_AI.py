from ball import Ball
from mastergoalGame import MastergoalGame
from player import Player
from position import Position
from collections import defaultdict, deque

class MinimaxAI:
    """Implementación de IA usando algoritmo Minimax con poda alfa-beta para Mastergoal."""
    
    def __init__(self, game, max_depth=1, evaluator=None):
        """
        Inicializa la IA con el juego y la profundidad máxima de búsqueda.
        Args:
            game: Instancia de MastergoalGame
            max_depth: Profundidad máxima de búsqueda 
            evaluator: Función de evaluación personalizada
        """
        self.game = game
        self.max_depth = max_depth
        self.evaluator = evaluator
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.position_history = defaultdict(lambda: deque(maxlen=4))  # por player_id
        self.game_state_history = deque(maxlen=8)  # Almacenar los últimos 8 estados (por ciclos)
        self.last_moves = deque(maxlen=4)  # Almacenar los últimos 4 movimientos (por ciclos)
        
    def get_state_hash(self, game):
        """
        Crea un hash del estado del juego para detectar ciclos.
        Args:
            game: Instancia de MastergoalGame
        Returns:
            String que representa el estado del juego
        """
        # Incluir posiciones de todos los jugadores y la pelota
        state = []
        # Posición de la pelota
        state.append(f"Ball:({game.ball.position.row},{game.ball.position.col})")
        # Posiciones de los jugadores, ordenados por ID para consistencia
        player_positions = []
        for player in sorted(game.players, key=lambda p: p.player_id):
            player_positions.append(f"P{player.player_id}:{player.team}:({player.position.row},{player.position.col})")
        state.extend(player_positions)
        # Incluir el equipo actual para distinguir entre estados similares con turno diferente
        state.append(f"Turn:{game.current_team}")
        # Unir todo en una cadena
        return "|".join(state)
        
    def get_best_move(self, team):
        """
        Encuentra y devuelve la mejor jugada para el equipo dado.
        Args:
            team: Equipo para el que se busca la mejor jugada (MastergoalGame.LEFT o MastergoalGame.RIGHT)
        Returns:
            La mejor jugada en formato (tipo, posición_origen, posición_destino)
        """
        # Reiniciar contadores para estadísticas
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # Guardar el estado original del juego
        original_state = self.game.get_game_state()
        
        # Asegurarse de que es el turno del equipo para el que buscamos la jugada
        if self.game.current_team != team:
            return None
        
        # Obtener el hash del estado actual
        current_state_hash = self.get_state_hash(self.game)
        # Actualizar el historial de estados
        self.game_state_history.append(current_state_hash)
        # Detectar ciclos
        is_in_cycle = len(self.game_state_history) == self.game_state_history.maxlen and \
                     len(set(self.game_state_history)) <= 3
        
        best_move = None
        best_value = float('-inf') if team == self.game.LEFT else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        all_moves = self.game.get_legal_moves()
        # To do, not yet: detectar ciclos y eliminar de mov valido tal vez?
        moves = all_moves
        for move in moves:
            # Crear una copia del juego para simular la jugada
            game_copy = self.clone_game(self.game)
            move_type, from_pos, to_pos = move
            if move_type == 'move':
                game_copy.execute_move(from_pos, to_pos)
            else:  # kick
                game_copy.execute_kick(to_pos)
                # Comprobar inmediatamente si la jugada resulta en un gol
                if (game_copy.is_goal_LEFT(to_pos)) or \
               (game_copy.is_goal_RIGHT(to_pos)):
                    # Si la jugada resulta en un gol, devolver esta jugada (Darle un valor?)
                    self.restore_game_state(self.game, original_state)
                    return move
                
            # Evaluar la jugada con minimax
            move_value = self.minimax(game_copy, self.max_depth - 1, alpha, beta, False if team == self.game.LEFT else True)
            
            # Actualizar la mejor jugada
            if team == self.game.LEFT:
                if move_value > best_value or (move_value == best_value and is_in_cycle and move not in self.last_moves):
                    best_value = move_value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if move_value < best_value or (move_value == best_value and is_in_cycle and move not in self.last_moves):
                    best_value = move_value
                    best_move = move
                beta = min(beta, best_value)

            # Poda alfa-beta
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        # To do, not yet: (ciclos) Guardar el movimiento elegido en el historial
              
        # Restaurar el estado original del juego
        self.restore_game_state(self.game, original_state)
        
        return best_move
        
    def minimax(self, game, depth, alpha, beta, is_maximizing):
        """
        Implementación del algoritmo minimax con poda alfa-beta.
        Args:
            game: Estado del juego a evaluar
            depth: Profundidad restante de búsqueda
            alpha: Valor alfa para poda
            beta: Valor beta para poda
            is_maximizing: True si es turno del jugador maximizador (LEFT)   
        Returns:
            Valor de la mejor jugada encontrada
        """
        self.nodes_evaluated += 1
        
        # Obtener el hash del estado actual para detección de ciclos locales
        state_hash = self.get_state_hash(game)
        
        # Verificar si el juego ha terminado o se alcanzó la profundidad máxima
        winner = game.get_winner()
        if winner == game.LEFT:
            return 10000  # Victoria para blanco
        elif winner == game.RIGHT:
            return -10000  # Victoria para rojo
        elif depth == 0:
            return self.evaluator.evaluate(game)
            
        moves = game.get_legal_moves()
        # No hay más jugadas posibles
        if not moves:
            return 0
            
        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                # Crear una copia del juego para simular la jugada
                game_copy = self.clone_game(game)
                
                # Ejecutar la jugada en la copia
                move_type, from_pos, to_pos = move
                if move_type == 'move':
                    game_copy.execute_move(from_pos, to_pos)
                else:  # kick
                    game_copy.execute_kick(to_pos)
                
                # Comprobar si esta jugada resulta en una victoria inmediata para el maximizador (LEFT)
                if game_copy.get_winner() == game.LEFT:
                    return 10000  # Devolver inmediatamente un valor muy alto
                    
                # Evaluar recursivamente
                eval_value = self.minimax(game_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_value)
                
                # Actualizar alfa
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                # Crear una copia del juego para simular la jugada
                game_copy = self.clone_game(game)
                
                # Ejecutar la jugada en la copia
                move_type, from_pos, to_pos = move
                if move_type == 'move':
                    game_copy.execute_move(from_pos, to_pos)
                else:  # kick
                    game_copy.execute_kick(to_pos)
                
                # Comprobar si esta jugada resulta en una victoria inmediata para el minimizador (RIGHT)
                if game_copy.get_winner() == game.RIGHT:
                    return -10000  # Devolver inmediatamente un valor muy bajo
                    
                # Evaluar recursivamente
                eval_value = self.minimax(game_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_value)
                
                # Actualizar beta
                beta = min(beta, eval_value)
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            return min_eval
    
    def track_player_position(self, player):
        """Registra la posición actual del jugador en su historial."""
        pos = (player.position.row, player.position.col)
        self.position_history[player.player_id].append(pos)

    def is_in_loop(self, player_id):
        history = self.position_history[player_id]
        return len(history) == history.maxlen and len(set(history)) <= 2

        
    def clone_game(self, game):
        """
        Crea una copia completa del estado del juego.
        
        Args:
            game: Instancia de MastergoalGame a clonar
            
        Returns:
            Nueva instancia de MastergoalGame con el mismo estado
        """
        # Crear un nuevo juego con el mismo nivel
        new_game = MastergoalGame(game.level)
        
        # Copiar el estado del juego
        new_game.left_goals = game.left_goals
        new_game.right_goals = game.right_goals
        new_game.current_team = game.current_team
        new_game.last_possession_team = game.last_possession_team
        new_game.passes_count = game.passes_count
        new_game.turn_count = game.turn_count
        new_game.skip_next_turn = game.skip_next_turn
        
        # Copiar la pelota
        new_game.ball = Ball(Position(game.ball.position.row, game.ball.position.col))
        
        # Copiar los jugadores
        new_game.players = []
        for player in game.players:
            new_player = Player(
                Position(player.position.row, player.position.col),
                player.team,
                player.player_id,
                player.is_goalkeeper
            )
            new_game.players.append(new_player)
            
        return new_game
        
    def restore_game_state(self, game, state):
        """
        Restaura el estado del juego a partir de un diccionario.
        
        Args:
            game: Instancia de MastergoalGame a modificar
            state: Diccionario con el estado a restaurar
        """
        game.level = state['level']
        game.left_goals = state['left_goals']
        game.right_goals = state['right_goals']
        game.current_team = state['current_team']
        game.last_possession_team = state.get('last_possession_team', game.LEFT)  # Valor por defecto si no existe
        game.passes_count = state['passes_count']
        game.turn_count = state['turn_count']
        game.skip_next_turn = state['skip_next_turn']
        
        # Restaurar la pelota
        ball_row, ball_col = state['ball_position']
        game.ball = Ball(Position(ball_row, ball_col))
        
        # Restaurar los jugadores
        game.players = []
        for player_data in state['players']:
            team, player_id, row, col, is_goalkeeper = player_data
            player = Player(Position(row, col), team, player_id, is_goalkeeper)
            game.players.append(player)