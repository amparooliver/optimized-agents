#!/usr/bin/env python3
"""
Agent Tournament System for Mastergoal game evaluation.
Allows controlled competition between different optimized agents with detailed analysis.
Perfect for thesis research and comparative evaluation.
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
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for a tournament agent."""
    name: str
    weights_file: str
    minimax_depth: Optional[int] = None
    description: str = ""
    
    # These will be loaded from the file
    level: Optional[int] = None
    weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        self.load_from_file()
    
    def load_from_file(self):
        """Load configuration from the weights file."""
        if not os.path.exists(self.weights_file):
            raise FileNotFoundError(f"Weights file not found: {self.weights_file}")
        
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
            
            # Extract level
            if 'level' in data:
                self.level = data['level']
            else:
                raise ValueError(f"No 'level' found in {self.weights_file}")
            
            # Extract minimax_depth if not specified
            if self.minimax_depth is None:
                if 'minimax_depth' in data:
                    self.minimax_depth = data['minimax_depth']
                else:
                    raise ValueError(f"No 'minimax_depth' found in {self.weights_file} and not specified in constructor")
            
            # Extract weights
            if 'weights' in data:
                self.weights = data['weights']
            elif 'best_individual' in data and 'weights' in data['best_individual']:
                self.weights = data['best_individual']['weights']
            else:
                raise ValueError(f"No weights found in {self.weights_file}")
            
            # Auto-generate description if not provided
            if not self.description:
                fitness = data.get('fitness', 'unknown')
                wins = data.get('wins', 'unknown')
                games = data.get('games_played', 'unknown')
                self.description = f"L{self.level}_D{self.minimax_depth}_F{fitness}_W{wins}/{games}"
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.weights_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {self.weights_file}: {e}")


@dataclass 
class GameResult:
    """Result of a single game."""
    agent1_name: str
    agent2_name: str
    winner: str  # 'agent1', 'agent2', or 'draw'
    score: float  # 1.0 for agent1 win, -1.0 for agent2 win, 0.0 for draw
    moves_count: int
    game_duration: float
    agent1_color: str  # 'LEFT' or 'RIGHT'
    agent2_color: str
    termination_reason: str  # 'goal', 'max_moves', 'error'


@dataclass
class MatchResult:
    """Result of a match (multiple games between two agents)."""
    agent1_name: str
    agent2_name: str
    games: List[GameResult]
    agent1_wins: int
    agent2_wins: int
    draws: int
    total_games: int
    agent1_score: float  # Total score for agent1
    agent2_score: float  # Total score for agent2
    match_duration: float
    
    @property
    def agent1_win_rate(self) -> float:
        return self.agent1_wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def agent2_win_rate(self) -> float:
        return self.agent2_wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games > 0 else 0.0


class TournamentManager:
    """Manages tournaments between different agents."""
    
    def __init__(self, 
                 output_dir: str = "tournament_results",
                 games_per_match: int = 10,
                 max_moves_per_game: int = 100,
                 time_limit_per_move: float = 5.0,
                 seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize tournament manager.
        
        Args:
            output_dir: Directory to save results
            games_per_match: Number of games in each match
            max_moves_per_game: Maximum moves before declaring draw
            time_limit_per_move: Time limit per move in seconds
            seed: Random seed for reproducibility
            verbose: Whether to print detailed progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.games_per_match = games_per_match
        self.max_moves_per_game = max_moves_per_game
        self.time_limit_per_move = time_limit_per_move
        self.verbose = verbose
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.setup_logging()
        
        # Tournament data
        self.agents: Dict[str, AgentConfig] = {}
        self.matches: List[MatchResult] = []
        self.tournament_start_time = None
        
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"tournament_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if self.verbose else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, config: AgentConfig):
        """Register an agent for the tournament."""
        if config.name in self.agents:
            self.logger.warning(f"Agent {config.name} already registered, overwriting")
        
        self.agents[config.name] = config
        self.logger.info(f"Registered agent: {config.name}")
        self.logger.info(f"  Level: {config.level}, Depth: {config.minimax_depth}")
        self.logger.info(f"  Weights file: {config.weights_file}")
        self.logger.info(f"  Description: {config.description}")
        
        if config.weights:
            self.logger.info(f"  Weights loaded: {len(config.weights)} features")
    
    def register_agent_from_file(self, name: str, weights_file: str, 
                                minimax_depth: Optional[int] = None, 
                                description: str = ""):
        """
        Convenience method to register an agent directly from a weights file.
        
        Args:
            name: Agent name
            weights_file: Path to the weights JSON file
            minimax_depth: Override minimax depth (if not in file)
            description: Override description (if not auto-generated)
        """
        config = AgentConfig(
            name=name,
            weights_file=weights_file,
            minimax_depth=minimax_depth,
            description=description
        )
        self.register_agent(config)

        return config
    
    def register_multiple_agents(self, agents_info: List[Dict[str, Any]]):
        """
        Register multiple agents from a list of dictionaries.
        
        Args:
            agents_info: List of dicts with keys: 'name', 'weights_file', 
                        and optionally 'minimax_depth', 'description'
        """
        for agent_info in agents_info:
            self.register_agent_from_file(**agent_info)
    
    def create_agent(self, config: AgentConfig):
        """Create an AI agent from configuration."""
        try:
            from minimax_AI import MinimaxAI
            from linear_evaluator import LinearEvaluator
            from mastergoalGame import MastergoalGame
            
            game = MastergoalGame(config.level)
            
            if config.weights:
                evaluator = LinearEvaluator(config.level, config.weights)
            else:
                evaluator = LinearEvaluator(config.level)  # Default weights
            
            agent = MinimaxAI(game, max_depth=config.minimax_depth, evaluator=evaluator)
            return agent
            
        except ImportError as e:
            self.logger.error(f"Failed to import requiRIGHT modules: {e}")
            raise
    
    def play_single_game(self, agent1_config: AgentConfig, agent2_config: AgentConfig, 
                        agent1_color: str) -> GameResult:
        """Play a single game between two agents."""
        try:
            from mastergoalGame import MastergoalGame
            
            # Create fresh game and agents
            game = MastergoalGame(agent1_config.level)
            agent1 = self.create_agent(agent1_config)
            agent2 = self.create_agent(agent2_config)
            
            # Set game references
            agent1.game = game
            agent2.game = game
            
            # Determine colors
            if agent1_color == 'LEFT':
                agent2_color = 'RIGHT'
                agents = {MastergoalGame.LEFT: agent1, MastergoalGame.RIGHT: agent2}
            else:
                agent2_color = 'LEFT' 
                agents = {MastergoalGame.RIGHT: agent1, MastergoalGame.LEFT: agent2}
            
            # Play the game
            start_time = time.time()
            moves_count = 0
            termination_reason = "max_moves"
            
            while not game.is_game_over() and moves_count < self.max_moves_per_game:
                current_team = game.current_team
                current_agent = agents[current_team]
                
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    termination_reason = "no_moves"
                    break
                
                # Get move with time limit
                move_start = time.time()
                try:
                    move = current_agent.get_best_move(current_team)
                    move_time = time.time() - move_start
                    
                    if move_time > self.time_limit_per_move:
                        self.logger.warning(f"Move took {move_time:.2f}s (limit: {self.time_limit_per_move}s)")
                    
                    if move not in legal_moves:
                        self.logger.warning(f"Invalid move selected, choosing random")
                        move = random.choice(legal_moves)
                    
                except Exception as e:
                    self.logger.error(f"Error getting move: {e}")
                    move = random.choice(legal_moves)
                
                # Execute move
                move_type, from_pos, to_pos = move
                if move_type == 'move':
                    game.execute_move(from_pos, to_pos)
                elif move_type == 'kick':
                    game.execute_kick(to_pos)
                
                moves_count += 1
            
            game_duration = time.time() - start_time
            
            # Determine result
            result = game.is_game_over()
            if result == 1:  # LEFT wins
                if agent1_color == 'LEFT':
                    winner = 'agent1'
                    score = 1.0
                else:
                    winner = 'agent2'
                    score = -1.0
                termination_reason = "goal"
            elif result == -1:  # RIGHT wins
                if agent1_color == 'RIGHT':
                    winner = 'agent1'
                    score = 1.0
                else:
                    winner = 'agent2'
                    score = -1.0
                termination_reason = "goal"
            else:  # Draw
                winner = 'draw'
                score = 0.0
            
            return GameResult(
                agent1_name=agent1_config.name,
                agent2_name=agent2_config.name,
                winner=winner,
                score=score,
                moves_count=moves_count,
                game_duration=game_duration,
                agent1_color=agent1_color,
                agent2_color=agent2_color,
                termination_reason=termination_reason
            )
            
        except Exception as e:
            self.logger.error(f"Error in game between {agent1_config.name} and {agent2_config.name}: {e}")
            return GameResult(
                agent1_name=agent1_config.name,
                agent2_name=agent2_config.name,
                winner='draw',
                score=0.0,
                moves_count=0,
                game_duration=0.0,
                agent1_color=agent1_color,
                agent2_color=agent2_color,
                termination_reason="error"
            )
    
    def play_match(self, agent1_name: str, agent2_name: str) -> MatchResult:
        """Play a complete match between two agents."""
        if agent1_name not in self.agents or agent2_name not in self.agents:
            raise ValueError(f"One or both agents not registeRIGHT")
        
        agent1_config = self.agents[agent1_name]
        agent2_config = self.agents[agent2_name]
        
        # Verify agents are same level
        if agent1_config.level != agent2_config.level:
            raise ValueError(f"Agents must be same level: {agent1_config.level} vs {agent2_config.level}")
        
        self.logger.info(f"Starting match: {agent1_name} vs {agent2_name}")
        self.logger.info(f"  {agent1_name}: Level {agent1_config.level}, Depth {agent1_config.minimax_depth}")
        self.logger.info(f"  {agent2_name}: Level {agent2_config.level}, Depth {agent2_config.minimax_depth}")
        
        match_start = time.time()
        games = []
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        
        # Play games, alternating colors
        for game_num in range(self.games_per_match):
            agent1_color = 'LEFT' if game_num % 2 == 0 else 'RIGHT'
            
            if self.verbose:
                self.logger.info(f"  Game {game_num + 1}/{self.games_per_match} - "
                               f"{agent1_name}({agent1_color}) vs {agent2_name}")
            
            game_result = self.play_single_game(agent1_config, agent2_config, agent1_color)
            games.append(game_result)
            
            if game_result.winner == 'agent1':
                agent1_wins += 1
            elif game_result.winner == 'agent2':
                agent2_wins += 1
            else:
                draws += 1
            
            if self.verbose:
                self.logger.info(f"    Result: {game_result.winner} ({game_result.moves_count} moves, "
                               f"{game_result.game_duration:.2f}s)")
        
        match_duration = time.time() - match_start
        
        # Calculate scores
        agent1_score = sum(1 for g in games if g.winner == 'agent1') + 0.5 * draws
        agent2_score = sum(1 for g in games if g.winner == 'agent2') + 0.5 * draws
        
        match_result = MatchResult(
            agent1_name=agent1_name,
            agent2_name=agent2_name,
            games=games,
            agent1_wins=agent1_wins,
            agent2_wins=agent2_wins,
            draws=draws,
            total_games=len(games),
            agent1_score=agent1_score,
            agent2_score=agent2_score,
            match_duration=match_duration
        )
        
        self.matches.append(match_result)
        
        self.logger.info(f"Match completed: {agent1_name} {agent1_wins}-{draws}-{agent2_wins} {agent2_name}")
        self.logger.info(f"  Score: {agent1_score:.1f} - {agent2_score:.1f}")
        self.logger.info(f"  Duration: {match_duration:.2f}s")
        
        return match_result
    
    def run_round_robin(self, agent_names: Optional[List[str]] = None) -> List[MatchResult]:
        """Run a round-robin tournament between specified agents."""
        if agent_names is None:
            agent_names = list(self.agents.keys())
        
        if len(agent_names) < 2:
            raise ValueError("Need at least 2 agents for tournament")
        
        self.logger.info(f"Starting round-robin tournament with {len(agent_names)} agents")
        self.logger.info(f"Agents: {', '.join(agent_names)}")
        
        self.tournament_start_time = time.time()
        tournament_matches = []
        
        # Play all combinations
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent1_name = agent_names[i]
                agent2_name = agent_names[j]
                
                match_result = self.play_match(agent1_name, agent2_name)
                tournament_matches.append(match_result)
        
        tournament_duration = time.time() - self.tournament_start_time
        self.logger.info(f"Tournament completed in {tournament_duration:.2f}s")
        
        return tournament_matches
    
    def generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of tournament results."""
        if not self.matches:
            return {}
        
        analysis = {
            'tournament_summary': {},
            'agent_performance': {},
            'head_to_head': {},
            'detailed_stats': {}
        }
        
        # Tournament summary
        total_games = sum(match.total_games for match in self.matches)
        total_duration = time.time() - self.tournament_start_time if self.tournament_start_time else 0
        
        analysis['tournament_summary'] = {
            'total_matches': len(self.matches),
            'total_games': total_games,
            'total_duration': total_duration,
            'average_game_duration': np.mean([g.game_duration for match in self.matches for g in match.games]),
            'average_moves_per_game': np.mean([g.moves_count for match in self.matches for g in match.games])
        }
        
        # Agent performance
        agent_stats = {}
        for agent_name in self.agents.keys():
            wins = 0
            losses = 0
            draws = 0
            total_score = 0
            games_played = 0
            
            for match in self.matches:
                if match.agent1_name == agent_name:
                    wins += match.agent1_wins
                    losses += match.agent2_wins
                    draws += match.draws
                    total_score += match.agent1_score
                    games_played += match.total_games
                elif match.agent2_name == agent_name:
                    wins += match.agent2_wins
                    losses += match.agent1_wins
                    draws += match.draws
                    total_score += match.agent2_score
                    games_played += match.total_games
            
            if games_played > 0:
                agent_stats[agent_name] = {
                    'wins': wins,
                    'losses': losses,
                    'draws': draws,
                    'games_played': games_played,
                    'win_rate': wins / games_played,
                    'loss_rate': losses / games_played,
                    'draw_rate': draws / games_played,
                    'score': total_score,
                    'score_rate': total_score / games_played,
                    'config': {
                        'level': self.agents[agent_name].level,
                        'depth': self.agents[agent_name].minimax_depth,
                        'description': self.agents[agent_name].description,
                        'weights_file': self.agents[agent_name].weights_file
                    }
                }
        
        analysis['agent_performance'] = agent_stats
        
        # Head-to-head results
        h2h = {}
        for match in self.matches:
            key = f"{match.agent1_name}_vs_{match.agent2_name}"
            h2h[key] = {
                'agent1': match.agent1_name,
                'agent2': match.agent2_name,
                'agent1_wins': match.agent1_wins,
                'agent2_wins': match.agent2_wins,
                'draws': match.draws,
                'agent1_score': match.agent1_score,
                'agent2_score': match.agent2_score,
                'agent1_win_rate': match.agent1_win_rate,
                'agent2_win_rate': match.agent2_win_rate
            }
        
        analysis['head_to_head'] = h2h
        
        return analysis
    
    def save_results(self, filename_prefix: str = "tournament") -> str:
        """Save tournament results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        # Save raw match results
        results_file = self.output_dir / f"{base_filename}.json"
        
        # Convert matches to serializable format
        matches_data = []
        for match in self.matches:
            match_data = {
                'agent1_name': match.agent1_name,
                'agent2_name': match.agent2_name,
                'agent1_wins': match.agent1_wins,
                'agent2_wins': match.agent2_wins,
                'draws': match.draws,
                'total_games': match.total_games,
                'agent1_score': match.agent1_score,
                'agent2_score': match.agent2_score,
                'match_duration': match.match_duration,
                'games': [
                    {
                        'agent1_name': g.agent1_name,
                        'agent2_name': g.agent2_name,
                        'winner': g.winner,
                        'score': g.score,
                        'moves_count': g.moves_count,
                        'game_duration': g.game_duration,
                        'agent1_color': g.agent1_color,
                        'agent2_color': g.agent2_color,
                        'termination_reason': g.termination_reason
                    }
                    for g in match.games
                ]
            }
            matches_data.append(match_data)
        
        # Include agent configurations
        agents_data = {}
        for name, config in self.agents.items():
            agents_data[name] = {
                'name': config.name,
                'level': config.level,
                'minimax_depth': config.minimax_depth,
                'weights_file': config.weights_file,
                'description': config.description,
                'weights': config.weights
            }
        
        results = {
            'tournament_config': {
                'games_per_match': self.games_per_match,
                'max_moves_per_game': self.max_moves_per_game,
                'time_limit_per_move': self.time_limit_per_move
            },
            'agents': agents_data,
            'matches': matches_data,
            'analysis': self.generate_analysis()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Generate CSV summary
        self.save_csv_summary(base_filename)
        
        return str(results_file)
    
    def save_csv_summary(self, base_filename: str):
        """Save summary tables as CSV files."""
        analysis = self.generate_analysis()
        
        if 'agent_performance' in analysis:
            # Agent performance table
            perf_data = []
            for agent_name, stats in analysis['agent_performance'].items():
                perf_data.append({
                    'Agent': agent_name,
                    'Level': stats['config']['level'],
                    'Depth': stats['config']['depth'],
                    'Games': stats['games_played'],
                    'Wins': stats['wins'],
                    'Losses': stats['losses'],
                    'Draws': stats['draws'],
                    'Win_Rate': f"{stats['win_rate']:.3f}",
                    'Score': f"{stats['score']:.1f}",
                    'Score_Rate': f"{stats['score_rate']:.3f}",
                    'Description': stats['config']['description'],
                    'Weights_File': stats['config']['weights_file']
                })
            
            perf_df = pd.DataFrame(perf_data)
            perf_df = perf_df.sort_values('Score_Rate', ascending=False)
            perf_file = self.output_dir / f"{base_filename}_performance.csv"
            perf_df.to_csv(perf_file, index=False)
            self.logger.info(f"Performance summary saved to {perf_file}")
        
        if 'head_to_head' in analysis:
            # Head-to-head results table
            h2h_data = []
            for match_key, stats in analysis['head_to_head'].items():
                h2h_data.append({
                    'Agent1': stats['agent1'],
                    'Agent2': stats['agent2'],
                    'Agent1_Wins': stats['agent1_wins'],
                    'Agent2_Wins': stats['agent2_wins'],
                    'Draws': stats['draws'],
                    'Agent1_Score': f"{stats['agent1_score']:.1f}",
                    'Agent2_Score': f"{stats['agent2_score']:.1f}",
                    'Agent1_WinRate': f"{stats['agent1_win_rate']:.3f}",
                    'Agent2_WinRate': f"{stats['agent2_win_rate']:.3f}"
                })
            
            h2h_df = pd.DataFrame(h2h_data)
            h2h_file = self.output_dir / f"{base_filename}_head_to_head.csv"
            h2h_df.to_csv(h2h_file, index=False)
            self.logger.info(f"Head-to-head results saved to {h2h_file}")
    
    def print_summary(self):
        """Print a summary of tournament results."""
        analysis = self.generate_analysis()
        
        print("\n" + "="*60)
        print("TOURNAMENT SUMMARY")
        print("="*60)
        
        if 'tournament_summary' in analysis:
            summary = analysis['tournament_summary']
            print(f"Total Matches: {summary['total_matches']}")
            print(f"Total Games: {summary['total_games']}")
            print(f"Total Duration: {summary['total_duration']:.2f}s")
            print(f"Avg Game Duration: {summary['average_game_duration']:.2f}s")
            print(f"Avg Moves per Game: {summary['average_moves_per_game']:.1f}")
        
        print("\n" + "-"*60)
        print("AGENT PERFORMANCE RANKING")
        print("-"*60)
        
        if 'agent_performance' in analysis:
            # Sort agents by score rate
            sorted_agents = sorted(
                analysis['agent_performance'].items(),
                key=lambda x: x[1]['score_rate'],
                reverse=True
            )
            
            print(f"{'Rank':<4} {'Agent':<20} {'Level':<5} {'Depth':<5} {'Games':<6} {'W-D-L':<8} {'Score':<7} {'Rate':<6}")
            print("-" * 60)
            
            for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
                wdl = f"{stats['wins']}-{stats['draws']}-{stats['losses']}"
                print(f"{rank:<4} {agent_name:<20} {stats['config']['level']:<5} "
                      f"{stats['config']['depth']:<5} {stats['games_played']:<6} "
                      f"{wdl:<8} {stats['score']:<7.1f} {stats['score_rate']:<6.3f}")
        
        print("\n" + "-"*60)
        print("HEAD-TO-HEAD RESULTS")
        print("-"*60)
        
        if 'head_to_head' in analysis:
            for match_key, stats in analysis['head_to_head'].items():
                print(f"{stats['agent1']} vs {stats['agent2']}: "
                      f"{stats['agent1_score']:.1f} - {stats['agent2_score']:.1f} "
                      f"({stats['agent1_wins']}-{stats['draws']}-{stats['agent2_wins']})")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Run agent tournaments')
    
    # Tournament parameters
    parser.add_argument('--games', type=int, default=2,
                       help='Games per match (default: 10)')
    parser.add_argument('--max-moves', type=int, default=100,
                       help='Maximum moves per game (default: 100)')
    parser.add_argument('--time-limit', type=float, default=15.0,
                       help='Time limit per move in seconds (default: 15.0)')
    parser.add_argument('--output-dir', type=str, default='tournament_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--quiet', action='store_true',
                       help='reduce verbose output')
    
    # Quick test option
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (2 games per match)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.games = 2
        print("Quick test mode: 2 games per match")
    
    # Create tournament manager
    tournament = TournamentManager(
        output_dir=args.output_dir,
        games_per_match=args.games,
        max_moves_per_game=args.max_moves,
        time_limit_per_move=args.time_limit,
        seed=args.seed,
        verbose=not args.quiet
    )
    print(f"Starting tournament with {len(tournament.agents)} agents")
    # Register agents with just file paths
    tournament.register_agent_from_file("Agent R1 D1 L1", "../Level1/Depth1/depth1_level1_run1/optimized_weights.json")
    tournament.register_agent_from_file("Agent R2 D1 L1", "../Level1/Depth1/depth1_level1_run2/optimized_weights.json") 
    tournament.register_agent_from_file("Agent R3 D1 L1", "../Level1/Depth1/depth1_level1_run3/optimized_weights.json")
    tournament.register_agent_from_file("Agent Alberto Default", "../Level1/Depth1/albertoDefaultL1D1.json")
    tournament.register_agent_from_file("Agent Alberto Average 10 Players", "../Level1/Depth1/albertoDefaultL1D1.json")
    #tournament.register_agent_from_file("Agent R1 D2 L1", "../Level1/Depth2/depth2_level1_run1/optimized_weights.json")
    #tournament.register_agent_from_file("Agent R2 D2 L1", "../Level1/Depth2/depth2_level1_run2/optimized_weights.json")
    #tournament.register_agent_from_file("Agent R3 D2 L1", "../Level1/Depth2/depth2_level1_run3/optimized_weights.json")
    #tournament.register_agent_from_file("Agent R1 D3 L1", "../Level1/Depth3/depth3_level1_run1/optimized_weights.json")
    # Run tournament
    matches = tournament.run_round_robin()
    tournament.print_summary()
    tournament.save_results()
    
    return tournament


# Example usage functions for convenience
def create_example_tournament():
    """Create an example tournament with sample agents."""
    tournament = TournamentManager(games_per_match=2, verbose=True)
    
    # Example agent registrations - replace with your actual weight files
    sample_agents = [
        {
            "name": "Optimized_Level1_Depth1",
            "weights_file": "optimized_weights_l1_d1.json",
            "description": "Level 1 optimized with depth 1"
        },
        {
            "name": "Optimized_Level1_Depth2", 
            "weights_file": "optimized_weights_l1_d2.json",
            "description": "Level 1 optimized with depth 2"
        },
        {
            "name": "Baseline_Level1",
            "weights_file": "baseline_weights_l1.json",
            "description": "Level 1 baseline agent"
        }
    ]
    
    print("Registering sample agents...")
    for agent_info in sample_agents:
        try:
            tournament.register_agent_from_file(**agent_info)
            print(f"✓ Registered {agent_info['name']}")
        except FileNotFoundError:
            print(f"✗ Could not find {agent_info['weights_file']}")
        except Exception as e:
            print(f"✗ Error with {agent_info['name']}: {e}")
    
    return tournament


def run_quick_test():
    """Run a quick test tournament."""
    print("Running quick test tournament...")
    
    tournament = TournamentManager(
        games_per_match=3,
        verbose=True,
        output_dir="test_results"
    )
    
    # Try to register the provided example agent
    try:
        tournament.register_agent_from_file(
            name="ExampleAgent",
            weights_file="optimized_weights.json"
        )
        print("✓ Successfully registered example agent")
        
        # Create a second agent for testing (you would replace this with actual files)
        # For demo purposes, we'll create another agent config manually
        demo_config = AgentConfig(
            name="DemoAgent",
            weights_file="optimized_weights.json",  # Using same file for demo
            minimax_depth=2,  # Different depth
            description="Demo agent with depth 2"
        )
        tournament.register_agent(demo_config)
        
        # Run tournament
        matches = tournament.run_round_robin()
        tournament.print_summary()
        result_file = tournament.save_results("quick_test")
        print(f"\nResults saved to: {result_file}")
        
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("Please ensure your weight files exist in the current directory")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    tournament = main()
    
    # Uncomment these lines to run examples:
    # tournament = create_example_tournament()
    # run_quick_test()