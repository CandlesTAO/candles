"""
SQLite-based storage for validator data including miner scores, scoring results, and performance history.

This module provides persistent storage for:
- Current miner scores and performance metrics
- Detailed scoring results for individual predictions
- Historical performance data over time
- Database statistics and analytics
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import bittensor


class SQLiteValidatorStorage:
    """
    SQLite-based storage implementation for validator data.

    Provides structured storage for:
    - Miner scores and metadata
    - Detailed scoring results
    - Performance history tracking
    - Database analytics and statistics
    """

    def __init__(self, config=None):
        """
        Initialize SQLite storage with database schema.

        Args:
            config: Validator configuration object
        """
        self.config = config

        # Determine database path
        if config and hasattr(config, 'sqlite_path') and config.sqlite_path is not None:
            db_dir = Path(config.sqlite_path)
        else:
            db_dir = Path.home() / ".candles" / "data"

        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / "validator_scores.db"

        # Initialize database
        self._initialize_database()

        bittensor.logging.info(f"SQLite storage initialized at: {self.db_path}")

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS miner_scores (
                    miner_uid INTEGER PRIMARY KEY,
                    score REAL NOT NULL,
                    hotkey TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS scoring_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interval_id TEXT NOT NULL,
                    prediction_id INTEGER NOT NULL,
                    miner_uid INTEGER NOT NULL,
                    color_score REAL NOT NULL,
                    price_score REAL NOT NULL,
                    confidence_weight REAL NOT NULL,
                    final_score REAL NOT NULL,
                    actual_color TEXT,
                    actual_price REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(prediction_id, miner_uid)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    miner_uid INTEGER NOT NULL,
                    score REAL NOT NULL,
                    average_score REAL,
                    prediction_count INTEGER,
                    color_accuracy REAL,
                    price_accuracy REAL,
                    days_since_registration INTEGER,
                    decay_adjusted_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better query performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoring_results_miner_uid
                ON scoring_results(miner_uid)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoring_results_interval_id
                ON scoring_results(interval_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_score_history_miner_uid
                ON score_history(miner_uid)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_score_history_timestamp
                ON score_history(timestamp)
            """)

            conn.commit()

    def save_miner_scores(self, miner_scores: dict[int, float], miner_hotkeys: dict[int, str]):
        """
        Save current miner scores to database.

        Args:
            miner_scores: Dictionary mapping miner UID to current score
            miner_hotkeys: Optional dictionary mapping miner UID to hotkey
        """
        if not miner_scores:
            return

        with sqlite3.connect(self.db_path) as conn:
            for miner_uid, score in miner_scores.items():
                hotkey = miner_hotkeys.get(miner_uid) if miner_hotkeys else None

                conn.execute("""
                    INSERT OR REPLACE INTO miner_scores (miner_uid, score, hotkey, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (miner_uid, score, hotkey, datetime.now(timezone.utc)))

            conn.commit()

        bittensor.logging.debug(f"Saved {len(miner_scores)} miner scores to SQLite")

    def save_scoring_results(self, scoring_results: dict[str, list[dict[str, Any]]]):
        """
        Save detailed scoring results to database.

        Args:
            scoring_results: Dictionary with interval_id as key and list of scoring results as value
        """
        if not scoring_results:
            return

        total_saved = 0
        with sqlite3.connect(self.db_path) as conn:
            for interval_id, results in scoring_results.items():
                for result in results:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO scoring_results (
                                interval_id, prediction_id, miner_uid, color_score,
                                price_score, confidence_weight, final_score,
                                actual_color, actual_price, timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            interval_id,
                            result.get('prediction_id'),
                            result.get('miner_uid'),
                            result.get('color_score'),
                            result.get('price_score'),
                            result.get('confidence_weight'),
                            result.get('final_score'),
                            result.get('actual_color'),
                            result.get('actual_price'),
                            datetime.now(timezone.utc)
                        ))
                        total_saved += 1
                    except Exception as e:
                        bittensor.logging.error(f"Error saving scoring result: {e}")
                        continue

            conn.commit()

        bittensor.logging.debug(f"Saved {total_saved} scoring results to SQLite")

    def save_score_history(self, miner_stats: dict[int, dict[str, Any]], days_since_registration: dict[int, int]):
        """
        Save miner performance history to database.

        Args:
            miner_stats: Dictionary with miner UID as key and performance stats as value
            days_since_registration: Optional dictionary mapping miner UID to days since registration
        """
        if not miner_stats:
            return

        with sqlite3.connect(self.db_path) as conn:
            for miner_uid, stats in miner_stats.items():
                days_reg = days_since_registration.get(miner_uid, 1) if days_since_registration else 1
                score = stats.get('score', 0.0)
                decay_adjusted = score / days_reg

                conn.execute("""
                    INSERT INTO score_history (
                        miner_uid, score, average_score, prediction_count,
                        color_accuracy, price_accuracy, days_since_registration,
                        decay_adjusted_score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    miner_uid,
                    score,
                    stats.get('average_score'),
                    stats.get('prediction_count'),
                    stats.get('color_accuracy'),
                    stats.get('price_accuracy'),
                    days_reg,
                    decay_adjusted,
                    datetime.now(timezone.utc)
                ))

            conn.commit()

        bittensor.logging.debug(f"Saved score history for {len(miner_stats)} miners to SQLite")

    def load_miner_scores(self) -> dict[int, float]:
        """
        Load current miner scores from database.

        Returns:
            Dictionary mapping miner UID to current score
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT miner_uid, score FROM miner_scores
                ORDER BY score DESC
            """)

            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_top_miners(self, limit: int = 10) -> list[tuple[int, float]]:
        """
        Get top performing miners by score.

        Args:
            limit: Maximum number of miners to return

        Returns:
            list of tuples (miner_uid, score) sorted by score descending
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT miner_uid, score FROM miner_scores
                ORDER BY score DESC
                LIMIT ?
            """, (limit,))

            return cursor.fetchall()

    def get_miner_performance_history(self, miner_uid: int, days: int = 30) -> list[dict[str, Any]]:
        """
        Get performance history for a specific miner.

        Args:
            miner_uid: Miner UID to query
            days: Number of days of history to retrieve

        Returns:
            list of performance records ordered by timestamp descending
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT score, average_score, prediction_count, color_accuracy,
                       price_accuracy, days_since_registration, decay_adjusted_score,
                       timestamp
                FROM score_history
                WHERE miner_uid = ?
                AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days), (miner_uid,))

            columns = ['score', 'average_score', 'prediction_count', 'color_accuracy',
                      'price_accuracy', 'days_since_registration', 'decay_adjusted_score', 'timestamp']

            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_scoring_results_by_interval(self, interval_id: str) -> list[dict[str, Any]]:
        """
        Get all scoring results for a specific interval.

        Args:
            interval_id: Interval ID to query

        Returns:
            list of scoring result dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT prediction_id, miner_uid, color_score, price_score,
                       confidence_weight, final_score, actual_color, actual_price, timestamp
                FROM scoring_results
                WHERE interval_id = ?
                ORDER BY final_score DESC
            """, (interval_id,))

            columns = ['prediction_id', 'miner_uid', 'color_score', 'price_score',
                      'confidence_weight', 'final_score', 'actual_color', 'actual_price', 'timestamp']

            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_database_stats(self) -> dict[str, int]:
        """
        Get database statistics including record counts.

        Returns:
            Dictionary with table names as keys and record counts as values
        """
        stats = {}

        with sqlite3.connect(self.db_path) as conn:
            # Get miner scores count
            cursor = conn.execute("SELECT COUNT(*) FROM miner_scores")
            stats['miner_scores'] = cursor.fetchone()[0]

            # Get scoring results count
            cursor = conn.execute("SELECT COUNT(*) FROM scoring_results")
            stats['scoring_results'] = cursor.fetchone()[0]

            # Get score history count
            cursor = conn.execute("SELECT COUNT(*) FROM score_history")
            stats['score_history'] = cursor.fetchone()[0]

            # Get unique miners in history
            cursor = conn.execute("SELECT COUNT(DISTINCT miner_uid) FROM score_history")
            stats['unique_miners_tracked'] = cursor.fetchone()[0]

            # Get date range of history
            cursor = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM score_history
                WHERE timestamp IS NOT NULL
            """)
            date_range = cursor.fetchone()
            if date_range[0] and date_range[1]:
                stats['history_date_range'] = f"{date_range[0]} to {date_range[1]}"

        return stats

    def cleanup_old_records(self, days_to_keep: int = 90):
        """
        Clean up old records to maintain database size.

        Args:
            days_to_keep: Number of days of history to retain
        """
        with sqlite3.connect(self.db_path) as conn:
            # Clean old score history
            cursor = conn.execute("""
                DELETE FROM score_history
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))

            deleted_history = cursor.rowcount

            # Clean old scoring results
            cursor = conn.execute("""
                DELETE FROM scoring_results
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))

            deleted_results = cursor.rowcount

            conn.commit()

        bittensor.logging.info(f"Cleaned up {deleted_history} history records and {deleted_results} scoring results older than {days_to_keep} days")

    def get_miner_score_trends(self, miner_uid: int, days: int = 7) -> dict[str, Any]:
        """
        Get score trends for a miner over time.

        Args:
            miner_uid: Miner UID to analyze
            days: Number of days to analyze

        Returns:
            Dictionary with trend analysis data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT decay_adjusted_score, timestamp
                FROM score_history
                WHERE miner_uid = ?
                AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp ASC
            """.format(days), (miner_uid,))

            scores_over_time = cursor.fetchall()

            if len(scores_over_time) < 2:
                return {'trend': 'insufficient_data', 'score_count': len(scores_over_time)}

            # Calculate trend
            first_score = scores_over_time[0][0]
            last_score = scores_over_time[-1][0]
            score_change = last_score - first_score

            # Calculate average and volatility
            scores = [s[0] for s in scores_over_time]
            avg_score = sum(scores) / len(scores)
            volatility = sum((s - avg_score) ** 2 for s in scores) / len(scores)

            return {
                'trend': 'improving' if score_change > 0 else 'declining' if score_change < 0 else 'stable',
                'score_change': score_change,
                'first_score': first_score,
                'last_score': last_score,
                'average_score': avg_score,
                'volatility': volatility,
                'score_count': len(scores_over_time)
            }

    def get_historical_daily_scores(self, miner_uid: int, days: int = 31) -> list[float]:
        """
        Get historical daily scores for a miner for score aggregation.

        This method retrieves the actual daily scores that should be summed
        according to the decay-and-scoring diagram workflow.

        Args:
            miner_uid: Miner UID to query
            days: Number of days to look back (default 31 as per diagram)

        Returns:
            list of daily scores (most recent first), up to the specified number of days
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get the most recent scores up to the specified limit,
            # regardless of exact date range to ensure proper 31-day capping
            cursor = conn.execute("""
                SELECT score, timestamp
                FROM score_history
                WHERE miner_uid = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (miner_uid, days))

            scores = [row[0] for row in cursor.fetchall()]

            bittensor.logging.debug(
                f"Retrieved {len(scores)} historical daily scores for miner {miner_uid} "
                f"over last {days} days: {scores[:5]}..." if len(scores) > 5 else f"over last {days} days: {scores}"
            )

            return scores

    def get_miner_days_since_first_score(self, miner_uid: int) -> int:
        """
        Get the number of days since the miner's first recorded score.

        This represents how long the miner has been actively scored,
        which should be used for the decay calculation denominator.

        Args:
            miner_uid: Miner UID to query

        Returns:
            Number of days since first score, minimum 1
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT MIN(timestamp) as first_score_date
                FROM score_history
                WHERE miner_uid = ?
                AND score IS NOT NULL
            """, (miner_uid,))

            result = cursor.fetchone()
            first_score_date = result[0] if result and result[0] else None

            if not first_score_date:
                bittensor.logging.debug(f"No score history found for miner {miner_uid}, defaulting to 1 day")
                return 1

            # Calculate days between first score and now
            from datetime import datetime, timezone
            first_date = datetime.fromisoformat(first_score_date.replace('Z', '+00:00'))
            current_date = datetime.now(timezone.utc)
            days_diff = (current_date - first_date).days

            # Ensure minimum of 1 day
            days_since_first = max(1, days_diff)

            bittensor.logging.debug(
                f"Miner {miner_uid} first score: {first_score_date}, "
                f"days since first score: {days_since_first}"
            )

            return days_since_first

    def clear_miner_history(self, miner_uid: int, old_hotkey: str):
        """
        Clear all historical data for a miner when their hotkey changes.

        This method removes:
        - Score history records
        - Scoring results
        - Current miner score entry

        Args:
            miner_uid: Miner UID to clear data for
            old_hotkey: Optional old hotkey for logging purposes
        """
        with sqlite3.connect(self.db_path) as conn:
            # Clear score history
            cursor = conn.execute("DELETE FROM score_history WHERE miner_uid = ?", (miner_uid,))
            deleted_history = cursor.rowcount

            # Clear scoring results
            cursor = conn.execute("DELETE FROM scoring_results WHERE miner_uid = ?", (miner_uid,))
            deleted_results = cursor.rowcount

            # Clear current miner scores
            cursor = conn.execute("DELETE FROM miner_scores WHERE miner_uid = ?", (miner_uid,))
            deleted_scores = cursor.rowcount

            conn.commit()

        hotkey_info = f" (old hotkey: {old_hotkey})" if old_hotkey else ""
        bittensor.logging.info(
            f"Cleared history for miner UID {miner_uid}{hotkey_info}: "
            f"{deleted_history} history records, {deleted_results} scoring results, "
            f"{deleted_scores} current score entries"
        )

    def get_last_prediction_id_for_miner(self, miner_uid: int) -> int | None:
        """
        Get the last prediction ID for a miner from scoring results.

        Args:
            miner_uid: Miner UID to query

        Returns:
            Last prediction ID or None if no records found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT prediction_id
                FROM scoring_results
                WHERE miner_uid = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (miner_uid,))

            result = cursor.fetchone()
            return result[0] if result else None
