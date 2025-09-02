#!/usr/bin/env python3
"""
Enhanced Agent Tournament System for Mastergoal game evaluation.
Optimized for thesis research with detailed logging and easy graph generation.
"""

import numpy as np
import random
import time
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for a tournament agent with enhanced metadata."""
    name: str
    weights_file: str
    minimax_depth: Optional[int] = None
    description: str = ""
    
    # Enhanced agent metadata
    level: Optional[int] = None
    weights: Optional[Dict[str, float]] = None
    generation: Optional[int] = None  # For evolutionary runs
    fitness_score: Optional[float] = None
    training_games: Optional[int] = None
    weight_source: str = ""  # e.g., "genetic_algorithm", "manual", "baseline"
    
    def __post_init__(self):
        self.load_from_file()
    
    def load_from_file(self):
        """Load configuration with enhanced metadata extraction."""
        if not os.path.exists(self.weights_file):
            raise FileNotFoundError(f"Weights file not found: {self.weights_file}")
        
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
            
            # Extract basic required fields
            self.level = data.get('level')
            if not self.level:
                raise ValueError(f"No 'level' found in {self.weights_file}")
            
            if self.minimax_depth is None:
                self.minimax_depth = data.get('minimax_depth')
                if not self.minimax_depth:
                    raise ValueError(f"No 'minimax_depth' found in {self.weights_file}")
            
            # Extract weights
            if 'weights' in data:
                self.weights = data['weights']
            elif 'best_individual' in data and 'weights' in data['best_individual']:
                self.weights = data['best_individual']['weights']
            else:
                raise ValueError(f"No weights found in {self.weights_file}")
            
            # Extract enhanced metadata
            self.generation = data.get('generation', data.get('best_individual', {}).get('generation'))
            self.fitness_score = data.get('fitness', data.get('best_individual', {}).get('fitness'))
            self.training_games = data.get('games_played', data.get('total_games'))
            
            # Determine weight source
            if 'genetic_algorithm' in self.weights_file.lower() or 'ga_' in self.weights_file.lower():
                self.weight_source = "genetic_algorithm"
            elif 'baseline' in self.weights_file.lower() or 'default' in self.weights_file.lower():
                self.weight_source = "baseline"
            elif 'manual' in self.weights_file.lower():
                self.weight_source = "manual"
            else:
                self.weight_source = "unknown"
            
            # Enhanced description generation
            if not self.description:
                parts = [f"L{self.level}", f"D{self.minimax_depth}"]
                if self.fitness_score is not None:
                    parts.append(f"F{self.fitness_score:.2f}")
                if self.generation is not None:
                    parts.append(f"G{self.generation}")
                if self.training_games is not None:
                    parts.append(f"T{self.training_games}")
                self.description = "_".join(parts)
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.weights_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {self.weights_file}: {e}")


@dataclass 
class EnhancedGameResult:
    """Enhanced game result with detailed timing and positional data."""
    # Basic game info
    agent1_name: str
    agent2_name: str
    winner: str  # 'agent1', 'agent2', or 'draw'
    score: float  # 1.0 for agent1 win, -1.0 for agent2 win, 0.0 for draw
    
    # Color assignments
    agent1_color: str  # 'LEFT' or 'RIGHT'
    agent2_color: str
    
    # Game flow metrics
    total_moves: int
    agent1_moves: int  # Number of moves made by agent1
    agent2_moves: int  # Number of moves made by agent2
    
    # Timing metrics
    game_duration: float
    agent1_total_time: float  # Total thinking time for agent1
    agent2_total_time: float  # Total thinking time for agent2
    agent1_avg_time: float    # Average time per move for agent1
    agent2_avg_time: float    # Average time per move for agent2
    agent1_max_time: float    # Longest single move time for agent1
    agent2_max_time: float    # Longest single move time for agent2
    
    # Game state info
    termination_reason: str  # 'goal', 'max_moves', 'error', 'timeout'
    final_ball_position: Tuple[int, int]  # Final ball coordinates
    
    # Move details for deeper analysis
    move_times: List[Tuple[str, float]]  # [(agent_name, move_time), ...]
    move_types: List[str]  # ['move', 'kick', 'move', ...]


@dataclass
class EnhancedMatchResult:
    """Enhanced match result with color-specific statistics."""
    agent1_name: str
    agent2_name: str
    games: List[EnhancedGameResult]
    
    # Overall match stats
    total_games: int
    match_duration: float
    
    # Win/loss breakdown
    agent1_wins: int
    agent2_wins: int
    draws: int
    agent1_score: float
    agent2_score: float
    
    # Color-specific performance
    agent1_wins_as_left: int
    agent1_wins_as_right: int
    agent2_wins_as_left: int
    agent2_wins_as_right: int
    draws_agent1_left: int
    draws_agent1_right: int
    
    # Performance metrics
    agent1_win_rate: float
    agent2_win_rate: float
    agent1_left_win_rate: float
    agent1_right_win_rate: float
    agent2_left_win_rate: float
    agent2_right_win_rate: float
    
    # Timing statistics
    avg_game_duration: float
    agent1_avg_move_time: float
    agent2_avg_move_time: float
    agent1_total_thinking_time: float
    agent2_total_thinking_time: float


class EnhancedTournamentManager:
    """Enhanced tournament manager with detailed logging and analysis."""
    
    def __init__(self, 
                 output_dir: str = "tournament_results",
                 games_per_match: int = 2,  # Exactly 2 games
                 max_moves_per_game: int = 1000,
                 time_limit_per_move: float = 60.0,
                 seed: Optional[int] = None,
                 verbose: bool = True):
        """Initialize enhanced tournament manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.games_per_match = games_per_match
        self.max_moves_per_game = max_moves_per_game
        self.time_limit_per_move = time_limit_per_move
        self.verbose = verbose
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.setup_enhanced_logging()
        
        # Tournament data
        self.agents: Dict[str, AgentConfig] = {}
        self.matches: List[EnhancedMatchResult] = []
        self.tournament_start_time = None
        
        # Enhanced tracking
        self.detailed_stats = {}
        self.move_history = []  # For detailed move analysis
        
    def setup_enhanced_logging(self):
        """Setup enhanced logging with structured output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create multiple log files for different aspects
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Main tournament log
        main_log = self.log_dir / f"tournament_{timestamp}.log"
        
        # Game details log
        self.game_log = self.log_dir / f"games_{timestamp}.log"
        
        # Performance metrics log
        self.perf_log = self.log_dir / f"performance_{timestamp}.log"
        
        # Setup main logger
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(main_log),
                logging.StreamHandler() if self.verbose else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup specialized loggers
        self.game_logger = logging.getLogger('games')
        game_handler = logging.FileHandler(self.game_log)
        game_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.game_logger.addHandler(game_handler)
        self.game_logger.setLevel(logging.INFO)
        
        self.perf_logger = logging.getLogger('performance')
        perf_handler = logging.FileHandler(self.perf_log)
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
    
    def register_agent(self, config: AgentConfig):
        """Register an agent with enhanced logging."""
        if config.name in self.agents:
            self.logger.warning(f"Agent {config.name} already registered, overwriting")
        
        self.agents[config.name] = config
        
        # Enhanced agent registration logging
        self.logger.info(f"=== AGENT REGISTRATION ===")
        self.logger.info(f"Name: {config.name}")
        self.logger.info(f"Level: {config.level}")
        self.logger.info(f"Minimax Depth: {config.minimax_depth}")
        self.logger.info(f"Weight Source: {config.weight_source}")
        self.logger.info(f"Weights File: {config.weights_file}")
        
        if config.fitness_score is not None:
            self.logger.info(f"Fitness Score: {config.fitness_score:.4f}")
        if config.generation is not None:
            self.logger.info(f"Generation: {config.generation}")
        if config.training_games is not None:
            self.logger.info(f"Training Games: {config.training_games}")
        
        self.logger.info(f"Description: {config.description}")
        
        if config.weights:
            self.logger.info(f"Feature Weights Loaded: {len(config.weights)} features")
            # Log key weight values for analysis
            weight_summary = {k: v for k, v in list(config.weights.items())[:5]}  # First 5 weights
            self.logger.info(f"Sample Weights: {weight_summary}")
    
    def register_agent_from_file(self, name: str, weights_file: str, 
                                minimax_depth: Optional[int] = None, 
                                description: str = ""):
        """Register agent from file with return for chaining."""
        config = AgentConfig(
            name=name,
            weights_file=weights_file,
            minimax_depth=minimax_depth,
            description=description
        )
        self.register_agent(config)
        return config
    
    def create_agent(self, config: AgentConfig):
        """Create an AI agent from configuration."""
        try:
            from minimax_AI import MinimaxAI
            from linear_evaluator import LinearEvaluator
            from mastergoalGame import MastergoalGame
            
            game = MastergoalGame(config.level)
            evaluator = LinearEvaluator(config.level, config.weights if config.weights else {})
            agent = MinimaxAI(game, max_depth=config.minimax_depth, evaluator=evaluator)
            return agent
            
        except ImportError as e:
            self.logger.error(f"Failed to import required modules: {e}")
            raise
    
    def play_enhanced_game(self, agent1_config: AgentConfig, agent2_config: AgentConfig, 
                          agent1_color: str) -> EnhancedGameResult:
        """Play a single game with enhanced data collection."""
        try:
            from mastergoalGame import MastergoalGame
            
            # Initialize game tracking
            game_start_time = time.time()
            move_times = []
            move_types = []
            agent_times = {agent1_config.name: [], agent2_config.name: []}
            agent_move_counts = {agent1_config.name: 0, agent2_config.name: 0}
            
            # Create game and agents
            game = MastergoalGame(agent1_config.level)
            agent1 = self.create_agent(agent1_config)
            agent2 = self.create_agent(agent2_config)
            
            agent1.game = game
            agent2.game = game
            
            # Set up color assignments
            if agent1_color == 'LEFT':
                agent2_color = 'RIGHT'
                agents = {MastergoalGame.LEFT: agent1, MastergoalGame.RIGHT: agent2}
                agent_names = {MastergoalGame.LEFT: agent1_config.name, MastergoalGame.RIGHT: agent2_config.name}
            else:
                agent2_color = 'LEFT'
                agents = {MastergoalGame.RIGHT: agent1, MastergoalGame.LEFT: agent2}
                agent_names = {MastergoalGame.RIGHT: agent1_config.name, MastergoalGame.LEFT: agent2_config.name}
            
            # Log game start
            self.game_logger.info(f"GAME START: {agent1_config.name}({agent1_color}) vs {agent2_config.name}({agent2_color})")
            
            # Play the game with detailed tracking
            total_moves = 0
            termination_reason = "max_moves"
            
            while not game.is_game_over() and total_moves < self.max_moves_per_game:
                current_team = game.current_team
                current_agent = agents[current_team]
                current_agent_name = agent_names[current_team]
                
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    termination_reason = "no_moves"
                    break
                
                # Track move timing
                move_start = time.time()
                try:
                    move = current_agent.get_best_move(current_team)
                    move_time = time.time() - move_start
                    
                    # Record timing data
                    agent_times[current_agent_name].append(move_time)
                    move_times.append((current_agent_name, move_time))
                    
                    if move_time > self.time_limit_per_move:
                        self.game_logger.warning(f"Move exceeded time limit: {move_time:.2f}s > {self.time_limit_per_move}s")
                        termination_reason = "timeout"
                    
                    if move not in legal_moves:
                        self.game_logger.warning(f"Invalid move selected by {current_agent_name}, choosing random")
                        move = random.choice(legal_moves)
                    
                except Exception as e:
                    self.game_logger.error(f"Error getting move from {current_agent_name}: {e}")
                    move = random.choice(legal_moves)
                    move_time = 0.1  # Minimal time for error case
                    agent_times[current_agent_name].append(move_time)
                    move_times.append((current_agent_name, move_time))
                
                # Execute move and track type
                move_type, from_pos, to_pos = move
                move_types.append(move_type)
                agent_move_counts[current_agent_name] += 1
                
                if move_type == 'move':
                    game.execute_move(from_pos, to_pos)
                elif move_type == 'kick':
                    game.execute_kick(to_pos)
                
                total_moves += 1
            
            game_duration = time.time() - game_start_time
            
            # Get final game state
            final_ball_pos = game.get_ball_position() if hasattr(game, 'get_ball_position') else (0, 0)
            
            # Determine winner and score
            result = game.is_game_over()
            if result == 1:  # LEFT wins
                winner = 'agent1' if agent1_color == 'LEFT' else 'agent2'
                score = 1.0 if agent1_color == 'LEFT' else -1.0
                termination_reason = "goal"
            elif result == -1:  # RIGHT wins
                winner = 'agent1' if agent1_color == 'RIGHT' else 'agent2'
                score = 1.0 if agent1_color == 'RIGHT' else -1.0
                termination_reason = "goal"
            else:  # Draw
                winner = 'draw'
                score = 0.0
            
            # Calculate timing statistics
            agent1_times = agent_times[agent1_config.name]
            agent2_times = agent_times[agent2_config.name]
            
            agent1_stats = {
                'total_time': sum(agent1_times),
                'avg_time': np.mean(agent1_times) if agent1_times else 0.0,
                'max_time': max(agent1_times) if agent1_times else 0.0,
                'moves': agent_move_counts[agent1_config.name]
            }
            
            agent2_stats = {
                'total_time': sum(agent2_times),
                'avg_time': np.mean(agent2_times) if agent2_times else 0.0,
                'max_time': max(agent2_times) if agent2_times else 0.0,
                'moves': agent_move_counts[agent2_config.name]
            }
            
            # Create enhanced result
            enhanced_result = EnhancedGameResult(
                agent1_name=agent1_config.name,
                agent2_name=agent2_config.name,
                winner=winner,
                score=score,
                agent1_color=agent1_color,
                agent2_color=agent2_color,
                total_moves=total_moves,
                agent1_moves=agent1_stats['moves'],
                agent2_moves=agent2_stats['moves'],
                game_duration=game_duration,
                agent1_total_time=agent1_stats['total_time'],
                agent2_total_time=agent2_stats['total_time'],
                agent1_avg_time=agent1_stats['avg_time'],
                agent2_avg_time=agent2_stats['avg_time'],
                agent1_max_time=agent1_stats['max_time'],
                agent2_max_time=agent2_stats['max_time'],
                termination_reason=termination_reason,
                final_ball_position=final_ball_pos,
                move_times=move_times,
                move_types=move_types
            )
            
            # Enhanced game logging
            self.game_logger.info(f"GAME RESULT: {winner} wins")
            self.game_logger.info(f"Duration: {game_duration:.2f}s, Moves: {total_moves}")
            self.game_logger.info(f"Agent1 ({agent1_config.name}): {agent1_stats['moves']} moves, "
                                f"avg {agent1_stats['avg_time']:.3f}s/move, total {agent1_stats['total_time']:.2f}s")
            self.game_logger.info(f"Agent2 ({agent2_config.name}): {agent2_stats['moves']} moves, "
                                f"avg {agent2_stats['avg_time']:.3f}s/move, total {agent2_stats['total_time']:.2f}s")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced game: {e}")
            # Return minimal error result
            return EnhancedGameResult(
                agent1_name=agent1_config.name,
                agent2_name=agent2_config.name,
                winner='draw',
                score=0.0,
                agent1_color=agent1_color,
                agent2_color=agent2_color,
                total_moves=0,
                agent1_moves=0,
                agent2_moves=0,
                game_duration=0.0,
                agent1_total_time=0.0,
                agent2_total_time=0.0,
                agent1_avg_time=0.0,
                agent2_avg_time=0.0,
                agent1_max_time=0.0,
                agent2_max_time=0.0,
                termination_reason="error",
                final_ball_position=(0, 0),
                move_times=[],
                move_types=[]
            )
    
    def play_enhanced_match(self, agent1_name: str, agent2_name: str) -> EnhancedMatchResult:
        """Play a complete match with enhanced statistics."""
        if agent1_name not in self.agents or agent2_name not in self.agents:
            raise ValueError(f"One or both agents not registered")
        
        agent1_config = self.agents[agent1_name]
        agent2_config = self.agents[agent2_name]
        
        if agent1_config.level != agent2_config.level:
            raise ValueError(f"Agents must be same level: {agent1_config.level} vs {agent2_config.level}")
        
        self.logger.info(f"=== MATCH START ===")
        self.logger.info(f"Agent1: {agent1_name} (L{agent1_config.level}, D{agent1_config.minimax_depth})")
        self.logger.info(f"Agent2: {agent2_name} (L{agent2_config.level}, D{agent2_config.minimax_depth})")
        
        match_start = time.time()
        games = []
        
        # Color-specific tracking
        agent1_wins_as_left = 0
        agent1_wins_as_right = 0
        agent2_wins_as_left = 0
        agent2_wins_as_right = 0
        draws_agent1_left = 0
        draws_agent1_right = 0
        
        # Play exactly 2 games (one as each color)
        for game_num in range(2):  # Fixed to 2 games
            agent1_color = 'LEFT' if game_num == 0 else 'RIGHT'
            
            self.logger.info(f"Game {game_num + 1}/2: {agent1_name}({agent1_color}) vs {agent2_name}")
            
            game_result = self.play_enhanced_game(agent1_config, agent2_config, agent1_color)
            games.append(game_result)
            
            # Track color-specific wins
            if game_result.winner == 'agent1':
                if agent1_color == 'LEFT':
                    agent1_wins_as_left += 1
                else:
                    agent1_wins_as_right += 1
            elif game_result.winner == 'agent2':
                if agent1_color == 'LEFT':  # agent2 is RIGHT
                    agent2_wins_as_right += 1
                else:  # agent2 is LEFT
                    agent2_wins_as_left += 1
            else:  # Draw
                if agent1_color == 'LEFT':
                    draws_agent1_left += 1
                else:
                    draws_agent1_right += 1
        
        match_duration = time.time() - match_start
        
        # Calculate match statistics
        agent1_wins = agent1_wins_as_left + agent1_wins_as_right
        agent2_wins = agent2_wins_as_left + agent2_wins_as_right
        draws = draws_agent1_left + draws_agent1_right
        
        agent1_score = agent1_wins + 0.5 * draws
        agent2_score = agent2_wins + 0.5 * draws
        
        # Calculate win rates
        total_games = len(games)
        agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0.0
        agent2_win_rate = agent2_wins / total_games if total_games > 0 else 0.0
        
        # Color-specific win rates (each agent plays 1 game as each color)
        agent1_left_win_rate = agent1_wins_as_left / 1.0
        agent1_right_win_rate = agent1_wins_as_right / 1.0
        agent2_left_win_rate = agent2_wins_as_left / 1.0  
        agent2_right_win_rate = agent2_wins_as_right / 1.0
        
        # Timing averages
        avg_game_duration = np.mean([g.game_duration for g in games])
        agent1_avg_move_time = np.mean([g.agent1_avg_time for g in games])
        agent2_avg_move_time = np.mean([g.agent2_avg_time for g in games])
        agent1_total_thinking_time = sum(g.agent1_total_time for g in games)
        agent2_total_thinking_time = sum(g.agent2_total_time for g in games)
        
        match_result = EnhancedMatchResult(
            agent1_name=agent1_name,
            agent2_name=agent2_name,
            games=games,
            total_games=total_games,
            match_duration=match_duration,
            agent1_wins=agent1_wins,
            agent2_wins=agent2_wins,
            draws=draws,
            agent1_score=agent1_score,
            agent2_score=agent2_score,
            agent1_wins_as_left=agent1_wins_as_left,
            agent1_wins_as_right=agent1_wins_as_right,
            agent2_wins_as_left=agent2_wins_as_left,
            agent2_wins_as_right=agent2_wins_as_right,
            draws_agent1_left=draws_agent1_left,
            draws_agent1_right=draws_agent1_right,
            agent1_win_rate=agent1_win_rate,
            agent2_win_rate=agent2_win_rate,
            agent1_left_win_rate=agent1_left_win_rate,
            agent1_right_win_rate=agent1_right_win_rate,
            agent2_left_win_rate=agent2_left_win_rate,
            agent2_right_win_rate=agent2_right_win_rate,
            avg_game_duration=avg_game_duration,
            agent1_avg_move_time=agent1_avg_move_time,
            agent2_avg_move_time=agent2_avg_move_time,
            agent1_total_thinking_time=agent1_total_thinking_time,
            agent2_total_thinking_time=agent2_total_thinking_time
        )
        
        self.matches.append(match_result)
        
        # Enhanced match logging
        self.logger.info(f"=== MATCH COMPLETED ===")
        self.logger.info(f"Final Score: {agent1_name} {agent1_score:.1f} - {agent2_score:.1f} {agent2_name}")
        self.logger.info(f"Color Performance - {agent1_name}: LEFT {agent1_wins_as_left}/1, RIGHT {agent1_wins_as_right}/1")
        self.logger.info(f"Color Performance - {agent2_name}: LEFT {agent2_wins_as_left}/1, RIGHT {agent2_wins_as_right}/1")
        self.logger.info(f"Average Move Times: {agent1_name} {agent1_avg_move_time:.5f}s, {agent2_name} {agent2_avg_move_time:.5f}s")
        self.logger.info(f"Match Duration: {match_duration:.4f}s")
        
        return match_result
    
    def run_round_robin(self, agent_names: Optional[List[str]] = None) -> List[EnhancedMatchResult]:
        """Run enhanced round-robin tournament."""
        if agent_names is None:
            agent_names = list(self.agents.keys())
        
        if len(agent_names) < 2:
            raise ValueError("Need at least 2 agents for tournament")
        
        self.logger.info(f"=== TOURNAMENT START ===")
        self.logger.info(f"Agents: {len(agent_names)} registered")
        self.logger.info(f"Format: {self.games_per_match} games per match (1 as LEFT, 1 as RIGHT)")
        self.logger.info(f"Max moves per game: {self.max_moves_per_game}")
        self.logger.info(f"Time limit per move: {self.time_limit_per_move}s")
        
        for name in agent_names:
            config = self.agents[name]
            self.logger.info(f"  {name}: L{config.level} D{config.minimax_depth} ({config.weight_source})")
        
        self.tournament_start_time = time.time()
        tournament_matches = []
        
        # Play all combinations
        total_matches = len(agent_names) * (len(agent_names) - 1) // 2
        match_count = 0
        
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                match_count += 1
                agent1_name = agent_names[i]
                agent2_name = agent_names[j]
                
                self.logger.info(f"Match {match_count}/{total_matches}: {agent1_name} vs {agent2_name}")
                
                match_result = self.play_enhanced_match(agent1_name, agent2_name)
                tournament_matches.append(match_result)
        
        tournament_duration = time.time() - self.tournament_start_time
        self.logger.info(f"=== TOURNAMENT COMPLETED ===")
        self.logger.info(f"Duration: {tournament_duration:.2f}s")
        self.logger.info(f"Total matches: {len(tournament_matches)}")
        self.logger.info(f"Total games: {sum(m.total_games for m in tournament_matches)}")
        
        return tournament_matches
    
    def generate_enhanced_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis optimized for thesis research."""
        if not self.matches:
            return {}
        
        analysis = {
            'tournament_info': {},
            'agent_profiles': {},
            'overall_performance': {},
            'color_analysis': {},
            'timing_analysis': {},
            'head_to_head': {},
            'statistical_summary': {}
        }
        
        # Tournament info
        total_games = sum(match.total_games for match in self.matches)
        total_duration = time.time() - self.tournament_start_time if self.tournament_start_time else 0
        
        analysis['tournament_info'] = {
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(self.matches),
            'total_games': total_games,
            'games_per_match': self.games_per_match,
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'avg_match_duration': np.mean([m.match_duration for m in self.matches]),
            'avg_game_duration': np.mean([g.game_duration for m in self.matches for g in m.games]),
            'max_moves_per_game': self.max_moves_per_game,
            'time_limit_per_move': self.time_limit_per_move
        }
        
        # Agent profiles with enhanced metadata
        for agent_name, config in self.agents.items():
            analysis['agent_profiles'][agent_name] = {
                'name': config.name,
                'level': config.level,
                'minimax_depth': config.minimax_depth,
                'weight_source': config.weight_source,
                'weights_file': config.weights_file,
                'description': config.description,
                'fitness_score': config.fitness_score,
                'generation': config.generation,
                'training_games': config.training_games,
                'total_features': len(config.weights) if config.weights else 0,
                'key_weights': dict(list(config.weights.items())[:10]) if config.weights else {}
            }
        
        # Overall performance analysis
        agent_performance = {}
        for agent_name in self.agents.keys():
            # Initialize counters
            total_games_played = 0
            total_wins = 0
            total_losses = 0
            total_draws = 0
            total_score = 0.0
            
            # Color-specific counters
            games_as_left = 0
            games_as_right = 0
            wins_as_left = 0
            wins_as_right = 0
            losses_as_left = 0
            losses_as_right = 0
            draws_as_left = 0
            draws_as_right = 0
            
            # Timing data
            total_thinking_time = 0.0
            all_move_times = []
            
            # Collect data from all matches
            for match in self.matches:
                if match.agent1_name == agent_name:
                    total_games_played += match.total_games
                    total_wins += match.agent1_wins
                    total_losses += match.agent2_wins
                    total_draws += match.draws
                    total_score += match.agent1_score
                    
                    # Color-specific data
                    games_as_left += 1  # Each match has 1 game as each color
                    games_as_right += 1
                    wins_as_left += match.agent1_wins_as_left
                    wins_as_right += match.agent1_wins_as_right
                    losses_as_left += (1 - match.agent1_wins_as_left - match.draws_agent1_left)
                    losses_as_right += (1 - match.agent1_wins_as_right - match.draws_agent1_right)
                    draws_as_left += match.draws_agent1_left
                    draws_as_right += match.draws_agent1_right
                    
                    # Timing data
                    total_thinking_time += match.agent1_total_thinking_time
                    all_move_times.extend([g.agent1_avg_time for g in match.games])
                    
                elif match.agent2_name == agent_name:
                    total_games_played += match.total_games
                    total_wins += match.agent2_wins
                    total_losses += match.agent1_wins
                    total_draws += match.draws
                    total_score += match.agent2_score
                    
                    # Color-specific data
                    games_as_left += 1
                    games_as_right += 1
                    wins_as_left += match.agent2_wins_as_left
                    wins_as_right += match.agent2_wins_as_right
                    losses_as_left += (1 - match.agent2_wins_as_left - match.draws_agent1_right)  # Note: draws are from agent1's perspective
                    losses_as_right += (1 - match.agent2_wins_as_right - match.draws_agent1_left)
                    draws_as_left += match.draws_agent1_right
                    draws_as_right += match.draws_agent1_left
                    
                    # Timing data
                    total_thinking_time += match.agent2_total_thinking_time
                    all_move_times.extend([g.agent2_avg_time for g in match.games])
            
            # Calculate performance metrics
            if total_games_played > 0:
                agent_performance[agent_name] = {
                    # Overall performance
                    'total_games': total_games_played,
                    'wins': total_wins,
                    'losses': total_losses,
                    'draws': total_draws,
                    'score': total_score,
                    'win_rate': total_wins / total_games_played,
                    'loss_rate': total_losses / total_games_played,
                    'draw_rate': total_draws / total_games_played,
                    'score_rate': total_score / total_games_played,
                    
                    # Color-specific performance
                    'games_as_left': games_as_left,
                    'games_as_right': games_as_right,
                    'wins_as_left': wins_as_left,
                    'wins_as_right': wins_as_right,
                    'losses_as_left': losses_as_left,
                    'losses_as_right': losses_as_right,
                    'draws_as_left': draws_as_left,
                    'draws_as_right': draws_as_right,
                    'left_win_rate': wins_as_left / games_as_left if games_as_left > 0 else 0,
                    'right_win_rate': wins_as_right / games_as_right if games_as_right > 0 else 0,
                    'color_advantage': (wins_as_left / games_as_left - wins_as_right / games_as_right) if games_as_left > 0 and games_as_right > 0 else 0,
                    
                    # Timing performance
                    'total_thinking_time': total_thinking_time,
                    'avg_thinking_time_per_game': total_thinking_time / total_games_played,
                    'avg_move_time': np.mean(all_move_times) if all_move_times else 0,
                    'move_time_std': np.std(all_move_times) if all_move_times else 0,
                    'min_move_time': np.min(all_move_times) if all_move_times else 0,
                    'max_move_time': np.max(all_move_times) if all_move_times else 0
                }
        
        analysis['overall_performance'] = agent_performance
        
        # Color analysis summary
        color_stats = {
            'left_advantage_summary': {},
            'right_advantage_summary': {},
            'color_balance_analysis': {}
        }
        
        # Calculate overall color advantages
        total_left_wins = sum(p['wins_as_left'] for p in agent_performance.values())
        total_right_wins = sum(p['wins_as_right'] for p in agent_performance.values())
        total_left_games = sum(p['games_as_left'] for p in agent_performance.values())
        total_right_games = sum(p['games_as_right'] for p in agent_performance.values())
        
        if total_left_games > 0 and total_right_games > 0:
            color_stats['color_balance_analysis'] = {
                'total_left_games': total_left_games,
                'total_right_games': total_right_games,
                'total_left_wins': total_left_wins,
                'total_right_wins': total_right_wins,
                'left_win_rate': total_left_wins / total_left_games,
                'right_win_rate': total_right_wins / total_right_games,
                'color_bias': (total_left_wins / total_left_games) - (total_right_wins / total_right_games),
                'color_bias_description': 'LEFT favored' if (total_left_wins / total_left_games) > (total_right_wins / total_right_games) else 'RIGHT favored'
            }
        
        analysis['color_analysis'] = color_stats
        
        # Timing analysis
        all_game_durations = [g.game_duration for m in self.matches for g in m.games]
        all_move_counts = [g.total_moves for m in self.matches for g in m.games]
        
        analysis['timing_analysis'] = {
            'game_duration_stats': {
                'mean': np.mean(all_game_durations),
                'std': np.std(all_game_durations),
                'min': np.min(all_game_durations),
                'max': np.max(all_game_durations),
                'median': np.median(all_game_durations)
            },
            'move_count_stats': {
                'mean': np.mean(all_move_counts),
                'std': np.std(all_move_counts),
                'min': np.min(all_move_counts),
                'max': np.max(all_move_counts),
                'median': np.median(all_move_counts)
            }
        }
        
        # Head-to-head detailed analysis
        h2h_detailed = {}
        for match in self.matches:
            key = f"{match.agent1_name}_vs_{match.agent2_name}"
            h2h_detailed[key] = {
                'agent1': match.agent1_name,
                'agent2': match.agent2_name,
                'agent1_score': match.agent1_score,
                'agent2_score': match.agent2_score,
                'agent1_wins': match.agent1_wins,
                'agent2_wins': match.agent2_wins,
                'draws': match.draws,
                
                # Color-specific results
                'agent1_wins_as_left': match.agent1_wins_as_left,
                'agent1_wins_as_right': match.agent1_wins_as_right,
                'agent2_wins_as_left': match.agent2_wins_as_left,
                'agent2_wins_as_right': match.agent2_wins_as_right,
                
                # Performance metrics
                'agent1_win_rate': match.agent1_win_rate,
                'agent2_win_rate': match.agent2_win_rate,
                'agent1_left_win_rate': match.agent1_left_win_rate,
                'agent1_right_win_rate': match.agent1_right_win_rate,
                'agent2_left_win_rate': match.agent2_left_win_rate,
                'agent2_right_win_rate': match.agent2_right_win_rate,
                
                # Timing comparison
                'agent1_avg_move_time': match.agent1_avg_move_time,
                'agent2_avg_move_time': match.agent2_avg_move_time,
                'speed_advantage': match.agent2_avg_move_time - match.agent1_avg_move_time,  # Positive means agent1 is faster
                
                # Match details
                'match_duration': match.match_duration,
                'avg_game_duration': match.avg_game_duration
            }
        
        analysis['head_to_head'] = h2h_detailed
        
        return analysis
    
    def save_enhanced_results(self, filename_prefix: str = "enhanced_tournament") -> Dict[str, str]:
        """Save enhanced results with multiple output formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        files_created = {}
        
        # 1. Main JSON results file
        results_file = self.output_dir / f"{base_filename}.json"
        
        # Convert to serializable format
        serializable_data = {
            'tournament_config': {
                'games_per_match': self.games_per_match,
                'max_moves_per_game': self.max_moves_per_game,
                'time_limit_per_move': self.time_limit_per_move,
                'output_directory': str(self.output_dir)
            },
            'agents': {name: asdict(config) for name, config in self.agents.items()},
            'raw_matches': [asdict(match) for match in self.matches],
            'analysis': self.generate_enhanced_analysis()
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        files_created['main_results'] = str(results_file)
        
        # 2. CSV files for easy analysis
        csv_files = self.save_enhanced_csv_files(base_filename)
        files_created.update(csv_files)
        
        # 3. Generate visualization plots
        plot_files = self.generate_analysis_plots(base_filename)
        files_created.update(plot_files)
        
        self.logger.info(f"Enhanced results saved:")
        for file_type, file_path in files_created.items():
            self.logger.info(f"  {file_type}: {file_path}")
        
        return files_created
    
    def save_enhanced_csv_files(self, base_filename: str) -> Dict[str, str]:
        """Save multiple CSV files for different aspects of analysis."""
        analysis = self.generate_enhanced_analysis()
        files_created = {}
        
        # 1. Overall agent performance table
        if 'overall_performance' in analysis:
            perf_data = []
            for agent_name, stats in analysis['overall_performance'].items():
                config = analysis['agent_profiles'][agent_name]
                perf_data.append({
                    'Agent_Name': agent_name,
                    'Level': config['level'],
                    'Minimax_Depth': config['minimax_depth'],
                    'Weight_Source': config['weight_source'],
                    'Fitness_Score': config.get('fitness_score', ''),
                    'Generation': config.get('generation', ''),
                    
                    # Overall performance
                    'Total_Games': stats['total_games'],
                    'Wins': stats['wins'],
                    'Losses': stats['losses'],
                    'Draws': stats['draws'],
                    'Score': f"{stats['score']:.1f}",
                    'Win_Rate': f"{stats['win_rate']:.3f}",
                    'Score_Rate': f"{stats['score_rate']:.3f}",
                    
                    # Color performance
                    'Wins_as_LEFT': stats['wins_as_left'],
                    'Wins_as_RIGHT': stats['wins_as_right'],
                    'LEFT_Win_Rate': f"{stats['left_win_rate']:.3f}",
                    'RIGHT_Win_Rate': f"{stats['right_win_rate']:.3f}",
                    'Color_Advantage': f"{stats['color_advantage']:.3f}",
                    
                    # Timing performance
                    'Avg_Move_Time': f"{stats['avg_move_time']:.4f}",
                    'Total_Thinking_Time': f"{stats['total_thinking_time']:.4f}",
                    'Move_Time_Std': f"{stats['move_time_std']:.4f}"
                })
            
            perf_df = pd.DataFrame(perf_data)
            perf_df = perf_df.sort_values('Score_Rate', ascending=False)
            perf_file = self.output_dir / f"{base_filename}_agent_performance.csv"
            perf_df.to_csv(perf_file, index=False)
            files_created['agent_performance'] = str(perf_file)
        
        # 2. Head-to-head results table
        if 'head_to_head' in analysis:
            h2h_data = []
            for match_key, stats in analysis['head_to_head'].items():
                h2h_data.append({
                    'Agent1': stats['agent1'],
                    'Agent2': stats['agent2'],
                    'Agent1_Score': f"{stats['agent1_score']:.1f}",
                    'Agent2_Score': f"{stats['agent2_score']:.1f}",
                    'Agent1_Wins': stats['agent1_wins'],
                    'Agent2_Wins': stats['agent2_wins'],
                    'Draws': stats['draws'],
                    
                    # Color-specific wins
                    'Agent1_Wins_as_LEFT': stats['agent1_wins_as_left'],
                    'Agent1_Wins_as_RIGHT': stats['agent1_wins_as_right'],
                    'Agent2_Wins_as_LEFT': stats['agent2_wins_as_left'],
                    'Agent2_Wins_as_RIGHT': stats['agent2_wins_as_right'],
                    
                    # Win rates
                    'Agent1_Win_Rate': f"{stats['agent1_win_rate']:.3f}",
                    'Agent2_Win_Rate': f"{stats['agent2_win_rate']:.3f}",
                    'Agent1_LEFT_Win_Rate': f"{stats['agent1_left_win_rate']:.3f}",
                    'Agent1_RIGHT_Win_Rate': f"{stats['agent1_right_win_rate']:.3f}",
                    
                    # Timing comparison
                    'Agent1_Avg_Move_Time': f"{stats['agent1_avg_move_time']:.4f}",
                    'Agent2_Avg_Move_Time': f"{stats['agent2_avg_move_time']:.4f}",
                    'Speed_Advantage_Agent1': f"{stats['speed_advantage']:.3f}",
                    
                    'Match_Duration': f"{stats['match_duration']:.5f}"
                })
            
            h2h_df = pd.DataFrame(h2h_data)
            h2h_file = self.output_dir / f"{base_filename}_head_to_head.csv"
            h2h_df.to_csv(h2h_file, index=False)
            files_created['head_to_head'] = str(h2h_file)
        
        # 3. Detailed game results
        game_data = []
        for match in self.matches:
            for game in match.games:
                game_data.append({
                    'Match_ID': f"{match.agent1_name}_vs_{match.agent2_name}",
                    'Agent1': game.agent1_name,
                    'Agent2': game.agent2_name,
                    'Agent1_Color': game.agent1_color,
                    'Agent2_Color': game.agent2_color,
                    'Winner': game.winner,
                    'Score': game.score,
                    'Total_Moves': game.total_moves,
                    'Agent1_Moves': game.agent1_moves,
                    'Agent2_Moves': game.agent2_moves,
                    'Game_Duration': f"{game.game_duration:.2f}",
                    'Agent1_Total_Time': f"{game.agent1_total_time:.5f}",
                    'Agent2_Total_Time': f"{game.agent2_total_time:.5f}",
                    'Agent1_Avg_Time': f"{game.agent1_avg_time:.5f}",
                    'Agent2_Avg_Time': f"{game.agent2_avg_time:.5f}",
                    'Agent1_Max_Time': f"{game.agent1_max_time:.5f}",
                    'Agent2_Max_Time': f"{game.agent2_max_time:.5f}",
                    'Termination_Reason': game.termination_reason,
                    'Final_Ball_Position': f"({game.final_ball_position[0]}, {game.final_ball_position[1]})"
                })
        
        game_df = pd.DataFrame(game_data)
        game_file = self.output_dir / f"{base_filename}_detailed_games.csv"
        game_df.to_csv(game_file, index=False)
        files_created['detailed_games'] = str(game_file)
        
        return files_created
    
    def generate_analysis_plots(self, base_filename: str) -> Dict[str, str]:
        """Generate visualization plots for thesis analysis."""
        analysis = self.generate_enhanced_analysis()
        files_created = {}
        
        # Set up matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        if 'overall_performance' not in analysis:
            return files_created
        
        # 1. Agent performance comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance data
        agents = list(analysis['overall_performance'].keys())
        win_rates = [analysis['overall_performance'][agent]['win_rate'] for agent in agents]
        score_rates = [analysis['overall_performance'][agent]['score_rate'] for agent in agents]
        left_win_rates = [analysis['overall_performance'][agent]['left_win_rate'] for agent in agents]
        right_win_rates = [analysis['overall_performance'][agent]['right_win_rate'] for agent in agents]
        
        # Win rates comparison
        x_pos = np.arange(len(agents))
        ax1.bar(x_pos, win_rates, alpha=0.8)
        ax1.set_xlabel('Agents')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Agent Win Rates')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(agents, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Score rates comparison
        ax2.bar(x_pos, score_rates, alpha=0.8, color='orange')
        ax2.set_xlabel('Agents')
        ax2.set_ylabel('Score Rate')
        ax2.set_title('Agent Score Rates')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(agents, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Color performance comparison
        width = 0.35
        ax3.bar(x_pos - width/2, left_win_rates, width, label='As LEFT', alpha=0.8)
        ax3.bar(x_pos + width/2, right_win_rates, width, label='As RIGHT', alpha=0.8)
        ax3.set_xlabel('Agents')
        ax3.set_ylabel('Win Rate by Color')
        ax3.set_title('Win Rate by Starting Color')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(agents, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Move time comparison
        move_times = [analysis['overall_performance'][agent]['avg_move_time'] for agent in agents]
        ax4.bar(x_pos, move_times, alpha=0.8, color='green')
        ax4.set_xlabel('Agents')
        ax4.set_ylabel('Average Move Time (s)')
        ax4.set_title('Agent Thinking Speed')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(agents, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        performance_plot = plots_dir / f"{base_filename}_performance_overview.png"
        plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
        plt.close()
        files_created['performance_overview'] = str(performance_plot)
        
        # 2. Color advantage analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Color advantage by agent
        color_advantages = [analysis['overall_performance'][agent]['color_advantage'] for agent in agents]
        colors = ['red' if x < 0 else 'blue' for x in color_advantages]
        
        ax1.barh(agents, color_advantages, color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Color Advantage (LEFT - RIGHT win rate)')
        ax1.set_title('Color Advantage by Agent\n(Positive = Better as LEFT)')
        ax1.grid(True, alpha=0.3)
        
        # Overall color distribution
        if 'color_analysis' in analysis and 'color_balance_analysis' in analysis['color_analysis']:
            color_data = analysis['color_analysis']['color_balance_analysis']
            left_wr = color_data['left_win_rate']
            right_wr = color_data['right_win_rate']
            
            ax2.bar(['LEFT', 'RIGHT'], [left_wr, right_wr], 
                   color=['blue', 'red'], alpha=0.7)
            ax2.set_ylabel('Overall Win Rate')
            ax2.set_title('Overall Color Performance')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Add bias information
            bias = color_data['color_bias']
            ax2.text(0.5, max(left_wr, right_wr) + 0.05, 
                    f"Bias: {bias:.3f}\n{color_data['color_bias_description']}", 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        color_plot = plots_dir / f"{base_filename}_color_analysis.png"
        plt.savefig(color_plot, dpi=300, bbox_inches='tight')
        plt.close()
        files_created['color_analysis'] = str(color_plot)
        
        # 3. Head-to-head heatmap
        if len(agents) > 1:
            # Create win rate matrix
            win_matrix = np.zeros((len(agents), len(agents)))
            for i, agent1 in enumerate(agents):
                for j, agent2 in enumerate(agents):
                    if i != j:
                        # Find the match
                        key1 = f"{agent1}_vs_{agent2}"
                        key2 = f"{agent2}_vs_{agent1}"
                        
                        if key1 in analysis['head_to_head']:
                            win_matrix[i][j] = analysis['head_to_head'][key1]['agent1_win_rate']
                        elif key2 in analysis['head_to_head']:
                            win_matrix[i][j] = analysis['head_to_head'][key2]['agent2_win_rate']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(win_matrix, annot=True, fmt='.3f', 
                       xticklabels=agents, yticklabels=agents,
                       cmap='RdYlBu_r', center=0.5, 
                       cbar_kws={'label': 'Win Rate'}, ax=ax)
            ax.set_title('Head-to-Head Win Rate Matrix\n(Row agent vs Column agent)')
            ax.set_xlabel('Opponent')
            ax.set_ylabel('Agent')
            
            plt.tight_layout()
            heatmap_plot = plots_dir / f"{base_filename}_head_to_head_heatmap.png"
            plt.savefig(heatmap_plot, dpi=300, bbox_inches='tight')
            plt.close()
            files_created['head_to_head_heatmap'] = str(heatmap_plot)
        
        # 4. Timing distribution analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Game duration distribution
        all_durations = [g.game_duration for m in self.matches for g in m.games]
        ax1.hist(all_durations, bins=15, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Game Duration (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Game Durations')
        ax1.grid(True, alpha=0.3)
        
        # Move count distribution
        all_moves = [g.total_moves for m in self.matches for g in m.games]
        ax2.hist(all_moves, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Total Moves per Game')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Game Lengths (Moves)')
        ax2.grid(True, alpha=0.3)
        
        # Move time by agent
        for i, agent in enumerate(agents):
            agent_move_times = []
            for match in self.matches:
                if match.agent1_name == agent:
                    agent_move_times.extend([g.agent1_avg_time for g in match.games])
                elif match.agent2_name == agent:
                    agent_move_times.extend([g.agent2_avg_time for g in match.games])
            
            ax3.hist(agent_move_times, bins=10, alpha=0.7, label=agent, edgecolor='black')
        
        ax3.set_xlabel('Average Move Time (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Move Times by Agent')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Termination reasons
        termination_counts = {}
        for match in self.matches:
            for game in match.games:
                reason = game.termination_reason
                termination_counts[reason] = termination_counts.get(reason, 0) + 1
        
        if termination_counts:
            reasons = list(termination_counts.keys())
            counts = list(termination_counts.values())
            
            ax4.pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Game Termination Reasons')
        
        plt.tight_layout()
        timing_plot = plots_dir / f"{base_filename}_timing_analysis.png"
        plt.savefig(timing_plot, dpi=300, bbox_inches='tight')
        plt.close()
        files_created['timing_analysis'] = str(timing_plot)
        
        self.logger.info(f"Generated {len(files_created)} analysis plots")
        return files_created
    
    def print_enhanced_summary(self):
        """Print comprehensive tournament summary optimized for thesis analysis."""
        analysis = self.generate_enhanced_analysis()
        
        print("\n" + "="*80)
        print("ENHANCED TOURNAMENT ANALYSIS SUMMARY")
        print("="*80)
        
        # Tournament info
        if 'tournament_info' in analysis:
            info = analysis['tournament_info']
            print(f"Tournament Duration: {info['total_duration_minutes']:.1f} minutes")
            print(f"Total Matches: {info['total_matches']}")
            print(f"Total Games: {info['total_games']} ({self.games_per_match} per match)")
            print(f"Average Game Duration: {info['avg_game_duration']:.2f}s")
            print(f"Games per Match: {info['games_per_match']} (1 as LEFT, 1 as RIGHT)")
        
        # Agent profiles
        print("\n" + "-"*80)
        print("AGENT PROFILES")
        print("-"*80)
        
        if 'agent_profiles' in analysis:
            for name, profile in analysis['agent_profiles'].items():
                print(f"\n{name}:")
                print(f"  Level: {profile['level']}, Depth: {profile['minimax_depth']}")
                print(f"  Source: {profile['weight_source']}")
                if profile.get('fitness_score'):
                    print(f"  Fitness: {profile['fitness_score']:.4f}")
                if profile.get('generation'):
                    print(f"  Generation: {profile['generation']}")
                print(f"  Features: {profile['total_features']}")
                print(f"  File: {profile['weights_file']}")
        
        # Overall performance ranking
        print("\n" + "-"*80)
        print("OVERALL PERFORMANCE RANKING")
        print("-"*80)
        
        if 'overall_performance' in analysis:
            # Sort by score rate
            sorted_agents = sorted(
                analysis['overall_performance'].items(),
                key=lambda x: x[1]['score_rate'],
                reverse=True
            )
            
            print(f"{'Rank':<4} {'Agent':<25} {'Games':<6} {'W-D-L':<8} {'Score':<7} {'Rate':<6} {'L_WR':<6} {'R_WR':<6} {'AvgTime':<8}")
            print("-" * 80)
            
            for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
                wdl = f"{stats['wins']}-{stats['draws']}-{stats['losses']}"
                print(f"{rank:<4} {agent_name:<25} {stats['total_games']:<6} {wdl:<8} "
                      f"{stats['score']:<7.1f} {stats['score_rate']:<6.3f} "
                      f"{stats['left_win_rate']:<6.3f} {stats['right_win_rate']:<6.3f} "
                      f"{stats['avg_move_time']:<8.3f}")
        
        # Color analysis
        print("\n" + "-"*80)
        print("COLOR ANALYSIS")
        print("-"*80)
        
        if 'color_analysis' in analysis and 'color_balance_analysis' in analysis['color_analysis']:
            color_data = analysis['color_analysis']['color_balance_analysis']
            print(f"Overall LEFT win rate: {color_data['left_win_rate']:.3f}")
            print(f"Overall RIGHT win rate: {color_data['right_win_rate']:.3f}")
            print(f"Color bias: {color_data['color_bias']:.3f} ({color_data['color_bias_description']})")
            
            # Show agents with strongest color preferences
            if 'overall_performance' in analysis:
                color_prefs = []
                for agent_name, stats in analysis['overall_performance'].items():
                    color_prefs.append((agent_name, stats['color_advantage']))
                
                color_prefs.sort(key=lambda x: abs(x[1]), reverse=True)
                print(f"\nStrongest color preferences:")
                for agent_name, advantage in color_prefs[:3]:
                    pref = "LEFT" if advantage > 0 else "RIGHT"
                    print(f"  {agent_name}: {advantage:.3f} (prefers {pref})")
        
        # Head-to-head highlights
        print("\n" + "-"*80)
        print("HEAD-TO-HEAD HIGHLIGHTS")
        print("-"*80)
        
        if 'head_to_head' in analysis:
            # Find most decisive matches
            decisive_matches = []
            for match_key, stats in analysis['head_to_head'].items():
                score_diff = abs(stats['agent1_score'] - stats['agent2_score'])
                decisive_matches.append((match_key, stats, score_diff))
            
            decisive_matches.sort(key=lambda x: x[2], reverse=True)
            
            print("Most decisive matches:")
            for match_key, stats, diff in decisive_matches[:5]:
                winner = stats['agent1'] if stats['agent1_score'] > stats['agent2_score'] else stats['agent2']
                print(f"  {stats['agent1']} vs {stats['agent2']}: "
                      f"{stats['agent1_score']:.1f}-{stats['agent2_score']:.1f} "
                      f"(Winner: {winner})")
                
                # Show color breakdown
                print(f"    {stats['agent1']}: LEFT {stats['agent1_wins_as_left']}/1, "
                      f"RIGHT {stats['agent1_wins_as_right']}/1")
                print(f"    {stats['agent2']}: LEFT {stats['agent2_wins_as_left']}/1, "
                      f"RIGHT {stats['agent2_wins_as_right']}/1")
        
        # Timing insights
        print("\n" + "-"*80)
        print("TIMING INSIGHTS")
        print("-"*80)
        
        if 'overall_performance' in analysis:
            # Fastest and slowest agents
            speed_ranking = sorted(
                analysis['overall_performance'].items(),
                key=lambda x: x[1]['avg_move_time']
            )
            
            print("Thinking speed ranking (fastest to slowest):")
            for rank, (agent_name, stats) in enumerate(speed_ranking, 1):
                print(f"  {rank}. {agent_name}: {stats['avg_move_time']:.3f}s avg, "
                      f"total {stats['total_thinking_time']:.1f}s")
        
        if 'timing_analysis' in analysis:
            timing = analysis['timing_analysis']
            print(f"\nGame duration: {timing['game_duration_stats']['mean']:.1f}s ± "
                  f"{timing['game_duration_stats']['std']:.1f}s")
            print(f"Moves per game: {timing['move_count_stats']['mean']:.1f} ± "
                  f"{timing['move_count_stats']['std']:.1f}")
        
        print("\n" + "="*80)


def main():
    """Enhanced main function with improved CLI."""
    parser = argparse.ArgumentParser(description='Enhanced Agent Tournament System for Thesis Research')
    
    # Tournament parameters
    parser.add_argument('--games', type=int, default=2,
                       help='Games per match - FIXED at 2 (1 as LEFT, 1 as RIGHT)')
    parser.add_argument('--max-moves', type=int, default=200,
                       help='Maximum moves per game (default: 1000)')
    parser.add_argument('--time-limit', type=float, default=400.0,
                       help='Time limit per move in seconds (default: 15.0)')
    parser.add_argument('--output-dir', type=str, default='enhanced_tournament_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbose output')
    
    # Agent registration options
    parser.add_argument('--agent-dir', type=str, default='.',
                       help='Directory to search for agent weight files')
    parser.add_argument('--auto-discover', action='store_true',
                       help='Automatically discover agent files in directory')
    
    # Analysis options
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--plots-only', action='store_true',
                       help='Only generate plots from existing results')
    
    # Quick test option
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode with sample agents')
    
    args = parser.parse_args()
    
    # Force games to 2 for thesis requirement
    if args.games != 2:
        print(f"Note: Games per match forced to 2 for thesis requirements (was {args.games})")
        args.games = 2
    
    # Create enhanced tournament manager
    tournament = EnhancedTournamentManager(
        output_dir=args.output_dir,
        games_per_match=args.games,
        max_moves_per_game=args.max_moves,
        time_limit_per_move=args.time_limit,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    print(f"Enhanced Tournament System for Thesis Research")
    print(f"Output directory: {args.output_dir}")
    print(f"Configuration: {args.games} games per match, {args.max_moves} max moves, {args.time_limit}s per move")
    
    if args.quick:
        print("\n=== QUICK TEST MODE ===")
        # Register sample agents for testing
        try:
            tournament.register_agent_from_file("Test_Agent_1", "optimized_weights.json")
            print("✓ Registered Test_Agent_1")
        except Exception as e:
            print(f"✗ Could not register test agent: {e}")
            return
    else:
        print("\n=== REGISTERING AGENTS ===")
        # Register your actual agents here
        agent_files = [
            #("Agent depth1_level1_run2", "../Level1/Depth1/depth1_level1_run2/optimized_weights.json"),
            #("Agent depth2_level1_run2", "../Level1/Depth2/depth2_level1_run2/optimized_weights.json"),
            #("Agent depth3_level1_run1", "../Level1/Depth3/depth3_level1_run1/optimized_weights.json"),
            #("Agent Alberto_Best_Agent_D1_L1", "../Level1/Depth1/albertoDefaultL1D1.json"),
            #("Agent Alberto_10_Avg_D1_L1", "../Level1/Depth1/albertoPromedioD1.json")
            # Uncomment above lines to include Level 1 agents
            #("Agent depth1_level2_run2", "../Level2/Depth1/depth1_level2_run2/optimized_weights.json"),
            #("Agent depth2_level2_run1", "../Level2/Depth2/depth2_level2_run1/optimized_weights.json"),
            #("Agent depth3_level2_run1", "../Level2/Depth3/depth3_level2_run1/optimized_weights.json"),
            #("Agent AlbertoBestAgent", "../Level2/Depth1/albertoBestAgent.json"),
            #("Agent AlbertoAvg", "../Level2/Depth1/albertoAvg.json"),
            # Uncomment above lines to include Level 2 agents
            ("Agent depth1_level3_run1", "../Level3/Depth1/depth1_level3_run1/optimized_weights.json"),
            ("Agent depth2_level3_run1", "../Level3/Depth2/depth2_level3_run1/optimized_weights.json"),
            ("Agent depth3_level3_run2", "../Level3/Depth3/depth3_level3_run2/optimized_weights.json"),
            ("Agent AlbertoBestAgentL3", "../Level3/Depth1/albertoBestAgentL3.json"),
            ("Agent AlbertoAvgL3", "../Level3/Depth1/albertoAvgL3.json")
            # Uncomment above lines to include Level 3 agents
        ]
        
        registered_count = 0
        for name, file_path in agent_files:
            try:
                tournament.register_agent_from_file(name, file_path)
                print(f"✓ Registered {name}")
                registered_count += 1
            except FileNotFoundError:
                print(f"✗ File not found: {file_path}")
            except Exception as e:
                print(f"✗ Error registering {name}: {e}")
        
        if registered_count < 2:
            print(f"Error: Need at least 2 agents, only registered {registered_count}")
            return
    
    print(f"\n=== STARTING TOURNAMENT ===")
    print(f"Registered agents: {len(tournament.agents)}")
    
    # Run tournament
    try:
        matches = tournament.run_round_robin()
        
        # Print enhanced summary
        tournament.print_enhanced_summary()
        
        # Save enhanced results
        print(f"\n=== SAVING RESULTS ===")
        files_created = tournament.save_enhanced_results()
        
        print(f"\nTournament completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Generated files:")
        for file_type, file_path in files_created.items():
            print(f"  - {file_type}: {Path(file_path).name}")
        
        return tournament
        
    except Exception as e:
        print(f"Error during tournament: {e}")
        tournament.logger.error(f"Tournament failed: {e}")
        return None


# Utility functions for thesis research
def analyze_existing_results(results_file: str):
    """Analyze results from a previous tournament run."""
    print(f"Analyzing existing results from: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        analysis = data.get('analysis', {})
        
        if 'overall_performance' in analysis:
            print("\nAgent Performance Summary:")
            sorted_agents = sorted(
                analysis['overall_performance'].items(),
                key=lambda x: x[1]['score_rate'],
                reverse=True
            )
            
            for rank, (name, stats) in enumerate(sorted_agents, 1):
                print(f"{rank}. {name}: {stats['score_rate']:.3f} score rate, "
                      f"{stats['win_rate']:.3f} win rate")
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return None


def compare_tournaments(file1: str, file2: str):
    """Compare results from two different tournaments."""
    print(f"Comparing tournaments:")
    print(f"  Tournament 1: {file1}")
    print(f"  Tournament 2: {file2}")
    
    try:
        with open(file1, 'r') as f:
            data1 = json.load(f)
        with open(file2, 'r') as f:
            data2 = json.load(f)
        
        analysis1 = data1.get('analysis', {})
        analysis2 = data2.get('analysis', {})
        
        # Compare common agents
        if 'overall_performance' in analysis1 and 'overall_performance' in analysis2:
            common_agents = set(analysis1['overall_performance'].keys()) & \
                           set(analysis2['overall_performance'].keys())
            
            print(f"\nComparison of {len(common_agents)} common agents:")
            print(f"{'Agent':<25} {'T1_Score':<10} {'T2_Score':<10} {'Difference':<10}")
            print("-" * 60)
            
            for agent in common_agents:
                score1 = analysis1['overall_performance'][agent]['score_rate']
                score2 = analysis2['overall_performance'][agent]['score_rate']
                diff = score2 - score1
                print(f"{agent:<25} {score1:<10.3f} {score2:<10.3f} {diff:<+10.3f}")
        
    except Exception as e:
        print(f"Error comparing tournaments: {e}")


if __name__ == "__main__":
    tournament = main()
    
    # Examples of additional analysis functions:
    # analyze_existing_results("enhanced_tournament_20241201_143022.json")
    # compare_tournaments("tournament1.json", "tournament2.json")