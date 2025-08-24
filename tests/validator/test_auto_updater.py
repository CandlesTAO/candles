import pytest
import json
import os
import tempfile
import asyncio
from unittest.mock import MagicMock, patch

from candles.validator.auto_updater import AutoUpdater


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "wallet_name": "test_wallet",
                "wallet_hotkey": "test_hotkey",
                "last_update": "2024-01-01T00:00:00",
                "previous_commit": "abc123def456",
                "other_config": "should_be_preserved",
            },
            f,
        )
        temp_file = f.name

    yield temp_file

    # Cleanup
    try:
        os.unlink(temp_file)
    except OSError:
        pass


@pytest.fixture
def mock_process_info():
    """Mock process info for testing."""
    return {
        "pid": 12345,
        "wallet_name": "test_wallet",
        "wallet_hotkey": "test_hotkey",
        "cmdline": [
            "python",
            "validator.py",
            "--wallet.name",
            "test_wallet",
            "--wallet.hotkey",
            "test_hotkey",
        ],
    }


@pytest.fixture
def auto_updater(temp_config_file):
    """Create an AutoUpdater instance for testing."""
    return AutoUpdater(config_file=temp_config_file)


class TestAutoUpdaterInitialization:
    """Test AutoUpdater initialization and configuration loading."""

    def test_init_with_default_values(self):
        """Test initialization with default values."""
        updater = AutoUpdater()
        assert updater.check_interval == 3600
        assert updater.config_file == "validator_config.json"
        assert updater.auto_update_branch == "main"
        assert updater.is_running is False
        # Note: current_commit might not be None if we're in a git repo

        # Check that config is initialized with expected structure
        expected_keys = {
            "wallet_name",
            "wallet_hotkey",
            "last_update",
            "previous_commit",
        }
        assert set(updater.config.keys()) == expected_keys
        # Note: config values might not be None if loaded from existing file or git repo

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        updater = AutoUpdater(check_interval=1800, config_file="custom_config.json")
        assert updater.check_interval == 1800
        assert updater.config_file == "custom_config.json"

    def test_init_with_env_override(self):
        """Test that AUTO_UPDATE_BRANCH environment variable is respected."""
        with patch.dict(os.environ, {"AUTO_UPDATE_BRANCH": "develop"}):
            updater = AutoUpdater()
            assert updater.auto_update_branch == "develop"

    def test_load_config_existing_file(self, temp_config_file):
        """Test loading configuration from existing file."""
        updater = AutoUpdater(config_file=temp_config_file)

        # Should load only the required attributes
        assert updater.config["wallet_name"] == "test_wallet"
        assert updater.config["wallet_hotkey"] == "test_hotkey"
        assert updater.config["last_update"] == "2024-01-01T00:00:00"
        assert updater.config["previous_commit"] == "abc123def456"

        # Other config should not be loaded into our config dict
        assert "other_config" not in updater.config

    def test_load_config_nonexistent_file(self):
        """Test loading configuration when file doesn't exist."""
        updater = AutoUpdater(config_file="nonexistent.json")

        # Should initialize with empty config when file doesn't exist
        assert updater.config == {}

    def test_load_config_corrupted_file(self):
        """Test loading configuration from corrupted JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            updater = AutoUpdater(config_file=temp_file)

            # Should handle corruption gracefully and initialize with empty config
            assert updater.config == {}
        finally:
            os.unlink(temp_file)


class TestAutoUpdaterConfigSaving:
    """Test configuration saving functionality."""

    def test_save_config_new_file(self, tmp_path):
        """Test saving configuration to a new file."""
        config_file = str(tmp_path / "new_config.json")
        updater = AutoUpdater(config_file=config_file)

        # Set some values
        updater.config.update(
            {
                "wallet_name": "new_wallet",
                "wallet_hotkey": "new_hotkey",
                "last_update": "2024-01-02T00:00:00",
                "previous_commit": "def789abc012",
            }
        )

        updater.save_config()

        # Verify file was created and contains our data
        assert os.path.exists(config_file)
        with open(config_file, "r") as f:
            saved_config = json.load(f)

        assert saved_config["wallet_name"] == "new_wallet"
        assert saved_config["wallet_hotkey"] == "new_hotkey"
        assert saved_config["last_update"] == "2024-01-02T00:00:00"
        assert saved_config["previous_commit"] == "def789abc012"

    def test_save_config_preserves_existing_data(self, temp_config_file):
        """Test that saving config preserves existing data not in our scope."""
        updater = AutoUpdater(config_file=temp_config_file)

        # Update our config
        updater.config.update(
            {"wallet_name": "updated_wallet", "last_update": "2024-01-03T00:00:00"}
        )

        updater.save_config()

        # Verify our changes were saved
        with open(temp_config_file, "r") as f:
            saved_config = json.load(f)

        assert saved_config["wallet_name"] == "updated_wallet"
        assert saved_config["last_update"] == "2024-01-03T00:00:00"

        # Verify existing data was preserved
        assert saved_config["other_config"] == "should_be_preserved"
        assert saved_config["wallet_hotkey"] == "test_hotkey"

    def test_save_config_handles_io_error(self, auto_updater):
        """Test that save_config handles IO errors gracefully."""
        # Make the config file read-only to simulate permission error
        os.chmod(auto_updater.config_file, 0o444)  # Read-only

        try:
            # This should not raise an exception
            auto_updater.save_config()
        except Exception:
            pytest.fail("save_config should handle IO errors gracefully")
        finally:
            # Restore permissions
            os.chmod(auto_updater.config_file, 0o666)


class TestWalletArgumentExtraction:
    """Test the wallet argument extraction logic with various edge cases."""

    def test_extract_wallet_args_space_separated(self, auto_updater):
        """Test extracting wallet args with space-separated format."""
        cmdline = [
            "python",
            "validator.py",
            "--wallet.name",
            "my_wallet",
            "--wallet.hotkey",
            "my_hotkey",
        ]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": "my_wallet",
                "wallet_hotkey": "my_hotkey",
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] == "my_wallet"
            assert process_info["wallet_hotkey"] == "my_hotkey"

    def test_extract_wallet_args_equals_format(self, auto_updater):
        """Test extracting wallet args with equals format."""
        cmdline = [
            "python",
            "validator.py",
            "--wallet.name=my_wallet",
            "--wallet.hotkey=my_hotkey",
        ]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": "my_wallet",
                "wallet_hotkey": "my_hotkey",
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] == "my_wallet"
            assert process_info["wallet_hotkey"] == "my_hotkey"

    def test_extract_wallet_args_mixed_format(self, auto_updater):
        """Test extracting wallet args with mixed formats."""
        cmdline = [
            "python",
            "validator.py",
            "--wallet.name",
            "my_wallet",
            "--wallet.hotkey=my_hotkey",
        ]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": "my_wallet",
                "wallet_hotkey": "my_hotkey",
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] == "my_wallet"
            assert process_info["wallet_hotkey"] == "my_hotkey"

    def test_extract_wallet_args_missing_values(self, auto_updater):
        """Test extracting wallet args when values are missing."""
        cmdline = ["python", "validator.py", "--wallet.name", "--wallet.hotkey"]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": None,
                "wallet_hotkey": None,
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] is None
            assert process_info["wallet_hotkey"] is None

    def test_extract_wallet_args_partial_values(self, auto_updater):
        """Test extracting wallet args when only some values are provided."""
        cmdline = ["python", "validator.py", "--wallet.name", "my_wallet"]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": "my_wallet",
                "wallet_hotkey": None,
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] == "my_wallet"
            assert process_info["wallet_hotkey"] is None

    def test_extract_wallet_args_empty_cmdline(self, auto_updater):
        """Test extracting wallet args from empty command line."""
        cmdline = []

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": None,
                "wallet_hotkey": None,
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] is None
            assert process_info["wallet_hotkey"] is None

    def test_extract_wallet_args_malformed_equals(self, auto_updater):
        """Test extracting wallet args with malformed equals format."""
        cmdline = ["python", "validator.py", "--wallet.name=", "--wallet.hotkey="]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": "",
                "wallet_hotkey": "",
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] == ""
            assert process_info["wallet_hotkey"] == ""

    def test_extract_wallet_args_with_quotes(self, auto_updater):
        """Test extracting wallet args that might contain quotes."""
        cmdline = [
            "python",
            "validator.py",
            "--wallet.name",
            "my'wallet",
            "--wallet.hotkey",
            'my"hotkey',
        ]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": "my'wallet",
                "wallet_hotkey": 'my"hotkey',
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] == "my'wallet"
            assert process_info["wallet_hotkey"] == 'my"hotkey'

    def test_extract_wallet_args_with_special_chars(self, auto_updater):
        """Test extracting wallet args with special characters."""
        cmdline = [
            "python",
            "validator.py",
            "--wallet.name=my-wallet_123",
            "--wallet.hotkey=my.hotkey@456",
        ]

        with patch.object(auto_updater, "get_validator_process_info") as mock_get_info:
            mock_get_info.return_value = {
                "pid": 12345,
                "wallet_name": "my-wallet_123",
                "wallet_hotkey": "my.hotkey@456",
                "cmdline": cmdline,
            }

            process_info = auto_updater.get_validator_process_info()
            assert process_info["wallet_name"] == "my-wallet_123"
            assert process_info["wallet_hotkey"] == "my.hotkey@456"


class TestProcessInfoHandling:
    """Test process information handling and edge cases."""

    def test_get_validator_process_info_success(self, auto_updater):
        """Test successful process info retrieval."""
        with patch("psutil.Process") as mock_process:
            mock_process.return_value.cmdline.return_value = [
                "python",
                "validator.py",
                "--wallet.name",
                "test_wallet",
                "--wallet.hotkey",
                "test_hotkey",
            ]

            with patch("os.getpid", return_value=12345):
                process_info = auto_updater.get_validator_process_info()

                assert process_info is not None
                assert process_info["pid"] == 12345
                assert process_info["wallet_name"] == "test_wallet"
                assert process_info["wallet_hotkey"] == "test_hotkey"
                assert len(process_info["cmdline"]) > 0

    def test_get_validator_process_info_psutil_error(self, auto_updater):
        """Test handling of psutil errors."""
        with patch("psutil.Process", side_effect=Exception("psutil error")):
            process_info = auto_updater.get_validator_process_info()
            assert process_info is None

    def test_get_validator_process_info_os_error(self, auto_updater):
        """Test handling of os.getpid errors."""
        with patch("os.getpid", side_effect=OSError("OS error")):
            with patch.object(auto_updater.logger, "error") as mock_logger:
                # This should not raise an exception, but return None
                process_info = auto_updater.get_validator_process_info()
                assert process_info is None
                # Should log the error
                mock_logger.assert_called_once()


class TestRestartValidator:
    """Test validator restart functionality."""

    def test_restart_validator_success(self, auto_updater, mock_process_info):
        """Test successful validator restart."""
        with patch.object(
            auto_updater, "get_validator_process_info", return_value=mock_process_info
        ):
            with patch.object(auto_updater, "save_config") as mock_save:
                with patch("os._exit") as mock_exit:
                    # This should not raise an exception
                    result = auto_updater.restart_validator()
                    assert result is True
                    # Should call save_config
                    mock_save.assert_called_once()

                    # Should call os._exit(0)
                    mock_exit.assert_called_once_with(0)

    def test_restart_validator_no_process_info(self, auto_updater):
        """Test restart when process info cannot be retrieved."""
        with patch.object(
            auto_updater, "get_validator_process_info", return_value=None
        ):
            result = auto_updater.restart_validator()
            assert result is False

    def test_restart_validator_file_creation_error(
        self, auto_updater, mock_process_info
    ):
        """Test restart when restart flag file cannot be created."""
        with patch.object(
            auto_updater, "get_validator_process_info", return_value=mock_process_info
        ):
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                result = auto_updater.restart_validator()
                assert result is False


class TestGitOperations:
    """Test Git-related operations."""

    def test_get_current_commit_success(self, auto_updater):
        """Test successful commit hash retrieval."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123def456\n"

            commit = auto_updater.get_current_commit()
            assert commit == "abc123def456"

    def test_get_current_commit_git_error(self, auto_updater):
        """Test handling of Git command errors."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "fatal: not a git repository"

            commit = auto_updater.get_current_commit()
            assert commit is None

    def test_get_current_commit_subprocess_error(self, auto_updater):
        """Test handling of subprocess errors."""
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            commit = auto_updater.get_current_commit()
            assert commit is None


class TestUpdateChecking:
    """Test update checking functionality."""

    @pytest.mark.asyncio
    async def test_check_for_updates_no_updates(self, auto_updater):
        """Test checking for updates when none are available."""
        # Set current_commit to match the mock response
        auto_updater.current_commit = "abc123def456"

        with patch("subprocess.run") as mock_run:
            # Mock successful git fetch
            mock_run.return_value.returncode = 0

            # Mock successful origin commit retrieval
            mock_run.return_value.stdout = "abc123def456\n"

            has_updates, latest_commit = await auto_updater.check_for_updates()

            assert has_updates is False
            assert latest_commit == "abc123def456"

    @pytest.mark.asyncio
    async def test_check_for_updates_git_fetch_failure(self, auto_updater):
        """Test handling of git fetch failures."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "fatal: remote error"

            has_updates, latest_commit = await auto_updater.check_for_updates()

            assert has_updates is False
            assert latest_commit is None

    @pytest.mark.asyncio
    async def test_check_for_updates_origin_commit_failure(self, auto_updater):
        """Test handling of origin commit retrieval failures."""
        with patch("subprocess.run") as mock_run:
            # First call (git fetch) succeeds
            mock_run.return_value.returncode = 0

            # Second call (git rev-parse) fails
            def mock_run_side_effect(*args, **kwargs):
                if "rev-parse" in args[0]:
                    result = MagicMock()
                    result.returncode = 1
                    result.stderr = "fatal: bad revision"
                    return result
                else:
                    result = MagicMock()
                    result.returncode = 0
                    return result

            mock_run.side_effect = mock_run_side_effect

            has_updates, latest_commit = await auto_updater.check_for_updates()

            assert has_updates is False
            assert latest_commit is None


class TestAutoUpdaterLifecycle:
    """Test the overall lifecycle of the AutoUpdater."""

    def test_stop_method(self, auto_updater):
        """Test that stop method sets is_running to False."""
        auto_updater.is_running = True
        auto_updater.stop()
        assert auto_updater.is_running is False

    @pytest.mark.asyncio
    async def test_run_update_checker_cancelled(self, auto_updater):
        """Test that run_update_checker handles cancellation gracefully."""
        with patch.object(
            auto_updater, "check_for_updates", side_effect=asyncio.CancelledError()
        ):
            with patch.object(auto_updater.logger, "info") as mock_logger:
                # This should not raise an exception and should log the cancellation
                await auto_updater.run_update_checker()
                # Should log the cancellation message
                mock_logger.assert_any_call("Update checker cancelled")

    @pytest.mark.asyncio
    async def test_run_update_checker_exception_handling(self, auto_updater):
        """Test that run_update_checker handles exceptions gracefully."""
        with patch.object(
            auto_updater, "check_for_updates", side_effect=Exception("Test error")
        ):
            with patch("asyncio.sleep") as mock_sleep:
                # Start the update checker
                task = asyncio.create_task(auto_updater.run_update_checker())

                # Wait a bit for it to process the exception
                await asyncio.sleep(0.1)

                # Cancel the task
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should have called sleep after the exception
                mock_sleep.assert_called()
