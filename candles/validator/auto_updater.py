#!/usr/bin/env python3
"""
GitHub Updater Module for Validator

This module handles automatic updates from GitHub by:
1. Periodically checking for new commits on the main branch
2. Pulling updates when available
3. Restarting the validator with the new code
4. Preserving wallet configuration during restarts
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime

import psutil


class AutoUpdater:
    """
    Handles automatic updates for the validator by monitoring the git repository.
    """

    def __init__(
        self,
        check_interval: int,
        config_file: str = "validator_config.json",
    ):
        """
        Initialize the GitHub updater.

        Args:
            check_interval: Interval in seconds to check for updates
            config_file: File to store current configuration
        """
        self.check_interval = check_interval
        self.config_file = config_file
        self.current_commit = None
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self.auto_update_branch = os.getenv("AUTO_UPDATE_BRANCH", "main")

        # Initialize config with only the attributes we need
        self.config = {
            "wallet_name": None,
            "wallet_hotkey": None,
            "last_update": None,
            "previous_commit": None,
        }

        self.load_config()

        self.current_commit = self.get_current_commit()
        if self.current_commit:
            self.logger.info(f"Current commit: {self.current_commit[:8]}")

    def load_config(self):
        """Load only the required validator configuration attributes from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    full_config = json.load(f)
                    # Only extract the specific attributes we need
                    self.config = {
                        "wallet_name": full_config.get("wallet_name"),
                        "wallet_hotkey": full_config.get("wallet_hotkey"),
                        "last_update": full_config.get("last_update"),
                        "previous_commit": full_config.get("previous_commit"),
                    }
                self.logger.info("Loaded required validator configuration attributes")
            else:
                self.config = {}
                self.logger.info("No existing configuration found, will create new one")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = {}

    def save_config(self):
        """Save only the required validator configuration attributes to file."""
        try:
            # Read existing config to preserve other attributes
            existing_config = {}
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, "r") as f:
                        existing_config = json.load(f)
                except Exception:
                    pass  # If we can't read existing config, start fresh

            # Update only our specific attributes
            existing_config.update(self.config)

            with open(self.config_file, "w") as f:
                json.dump(existing_config, f, indent=2)
            self.logger.info("Saved required validator configuration attributes")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def get_current_commit(self) -> str | None:
        """Get the current commit hash of the repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.logger.warning("Failed to get current commit hash")
                return None
        except Exception as e:
            self.logger.error(f"Error getting current commit: {e}")
            return None

    async def check_for_updates(self) -> tuple[bool, str | None]:
        """
        Check if there are updates available by comparing local and remote commits.

        Returns:
            Tuple of (has_updates, latest_commit_hash)
        """
        try:
            # Fetch latest changes from origin
            result = subprocess.run(
                ["git", "fetch", "origin"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )

            if result.returncode != 0:
                self.logger.warning(f"Git fetch failed: {result.stderr}")
                return False, None

            result = subprocess.run(
                ["git", "rev-parse", f"origin/{self.auto_update_branch}"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"Failed to get origin/{self.auto_update_branch} commit: {result.stderr}"
                )
                return False, None

            latest_commit = result.stdout.strip()

            if self.current_commit and latest_commit != self.current_commit:
                self.logger.info(
                    f"Update available: {self.current_commit[:8]} -> {latest_commit[:8]}"
                )
                return True, latest_commit
            else:
                return False, latest_commit

        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
            return False, None

    async def pull_updates(self) -> bool:
        """
        Pull the latest updates from GitHub.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Pulling latest updates from GitHub...")

            # Preserve important files before git reset
            important_files = [
                "start_validator.sh",
                "pm2_start_validator.sh",
                ".env",
                "validator_config.json",
            ]

            preserved_files = {}
            for file_path in important_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            preserved_files[file_path] = f.read()
                        self.logger.info(f"Preserved {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not preserve {file_path}: {e}")

            # Fetch latest changes
            result = subprocess.run(
                ["git", "fetch", "origin"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )

            if result.returncode != 0:
                self.logger.error(f"Git fetch failed: {result.stderr}")
                return False

            result = subprocess.run(
                ["git", "reset", "--hard", f"origin/{self.auto_update_branch}"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )

            if result.returncode != 0:
                self.logger.error(f"Git reset failed: {result.stderr}")
                return False

            # Restore preserved files if they were overwritten
            for file_path, content in preserved_files.items():
                try:
                    # Check if the file was modified by git reset
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        with open(file_path, "w") as f:
                            f.write(content)
                        self.logger.info(f"Restored {file_path}")
                        # Make executable if it's a shell script
                        if file_path.endswith(".sh"):
                            os.chmod(file_path, 0o755)
                except Exception as e:
                    self.logger.warning(f"Could not restore {file_path}: {e}")

            # Update current commit
            self.current_commit = self.get_current_commit()
            self.logger.info(
                f"Successfully updated to commit: {self.current_commit[:8]}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error pulling updates: {e}")
            return False

    def get_validator_process_info(self) -> dict | None:
        """
        Get information about the current validator process.

        Returns:
            Process info dict or None if not found
        """
        try:
            current_pid = os.getpid()
            process = psutil.Process(current_pid)

            # Get command line arguments
            cmdline = process.cmdline()

            def extract_wallet_args(cmdline):
                wallet_name = None
                wallet_hotkey = None
                for i, arg in enumerate(cmdline):
                    if arg == "--wallet.name" and i + 1 < len(cmdline):
                        wallet_name = cmdline[i + 1]
                    elif arg.startswith("--wallet.name="):
                        wallet_name = arg.split("=", 1)[1]
                    if arg == "--wallet.hotkey" and i + 1 < len(cmdline):
                        wallet_hotkey = cmdline[i + 1]
                    elif arg.startswith("--wallet.hotkey="):
                        wallet_hotkey = arg.split("=", 1)[1]
                return wallet_name, wallet_hotkey

            wallet_name, wallet_hotkey = extract_wallet_args(cmdline)

            return {
                "pid": current_pid,
                "wallet_name": wallet_name,
                "wallet_hotkey": wallet_hotkey,
                "cmdline": cmdline,
            }
        except Exception as e:
            self.logger.error(f"Error getting process info: {e}")
            return None

    def restart_validator(self):
        """
        Restart the validator by creating a restart flag file.
        The shell script will detect this and restart the validator.
        """
        try:
            self.logger.info("Restarting validator...")
            process_info = self.get_validator_process_info()
            if not process_info:
                self.logger.error("Could not get process info for restart")
                return False

            # Save current configuration
            self.config.update(
                {
                    "wallet_name": process_info["wallet_name"],
                    "wallet_hotkey": process_info["wallet_hotkey"],
                    "last_update": datetime.now().isoformat(),
                    "previous_commit": self.current_commit,
                }
            )
            self.save_config()

            # Create restart flag file
            restart_flag_file = ".validator_restart"
            try:
                with open(restart_flag_file, "w") as f:
                    f.write(f"Restart requested at {datetime.now().isoformat()}\n")
                    f.write(f"Previous commit: {self.current_commit}\n")
                    f.write(f"Wallet: {process_info['wallet_name']}\n")
                    f.write(f"Hotkey: {process_info['wallet_hotkey']}\n")

                self.logger.info(f"Created restart flag file: {restart_flag_file}")

                # Exit current process gracefully
                self.logger.info("Exiting current validator process for restart")
                os._exit(0)

            except Exception as e:
                self.logger.error(f"Failed to create restart flag file: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error restarting validator: {e}")
            return False

    async def run_update_checker(self):
        """
        Main loop for checking and applying updates.
        """
        self.is_running = True
        self.logger.info("Starting GitHub update checker")

        while self.is_running:
            try:
                # Check for updates
                has_updates, latest_commit = await self.check_for_updates()

                if has_updates:
                    self.logger.info("Updates detected, pulling latest code...")

                    # Pull updates
                    if await self.pull_updates():
                        self.logger.info(
                            "Updates pulled successfully, restarting validator..."
                        )

                        # Small delay to ensure files are written
                        await asyncio.sleep(2)

                        # Restart validator
                        self.restart_validator()
                        break
                    else:
                        self.logger.error("Failed to pull updates")

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                self.logger.info("Update checker cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in update checker: {e}")
                await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the update checker."""
        self.is_running = False
        self.logger.info("GitHub update checker stopped")


async def main():
    """Test function for the auto-updater."""
    updater = AutoUpdater(
        check_interval=int(os.getenv("AUTO_UPDATE_INTERVAL", 43200)),
        config_file=os.getenv("VALIDATOR_CONFIG_FILE", "validator_config.json"),
    )
    await updater.run_update_checker()


if __name__ == "__main__":
    asyncio.run(main())
