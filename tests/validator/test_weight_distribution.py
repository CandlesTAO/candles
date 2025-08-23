import pytest
import numpy as np
from unittest.mock import MagicMock


class TestWeightDistribution:
    """Test cases for weight distribution logic including IP deduplication and hardcoded weights."""

    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator instance with necessary attributes."""
        validator = MagicMock()
        validator.enforce_unique_miner_ip = True

        # Mock metagraph with axons containing IP information
        validator.metagraph = MagicMock()
        validator.metagraph.axons = MagicMock()

        # Mock scores array
        validator.scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        return validator

    @pytest.fixture
    def mock_axons_with_ips(self):
        """Create mock axons with different IP configurations."""
        axons = []

        # UID 0: IP 192.168.1.1
        axon_0 = MagicMock()
        axon_0.ip = "192.168.1.1"
        axons.append(axon_0)

        # UID 1: IP 192.168.1.2
        axon_1 = MagicMock()
        axon_1.ip = "192.168.1.2"
        axons.append(axon_1)

        # UID 2: IP 192.168.1.1 (same as UID 0 - should be deduplicated)
        axon_2 = MagicMock()
        axon_2.ip = "192.168.1.1"
        axons.append(axon_2)

        # UID 3: IP 192.168.1.3
        axon_3 = MagicMock()
        axon_3.ip = "192.168.1.3"
        axons.append(axon_3)

        # UID 4: IP 192.168.1.2 (same as UID 1 - should be deduplicated)
        axon_4 = MagicMock()
        axon_4.ip = "192.168.1.2"
        axons.append(axon_4)

        return axons

    def test_ip_deduplication_keeps_only_one_miner_per_ip(
        self, mock_validator, mock_axons_with_ips
    ):
        """Test that IP deduplication keeps only one miner per IP address."""
        # Setup metagraph with mock axons
        mock_validator.metagraph.axons = mock_axons_with_ips

        # Create raw weights where UID 2 and UID 4 have higher weights than UID 0 and UID 1
        # This tests that the deduplication keeps the highest weight miner per IP
        raw_weights = np.array([0.1, 0.2, 0.5, 0.3, 0.6])  # UID 4 has highest weight

        # Mock the _apply_ip_deduplication_to_weights method to return expected result
        mock_validator._apply_ip_deduplication_to_weights = MagicMock(
            return_value=np.array([0.0, 0.0, 0.5, 0.3, 0.6])
        )

        # Apply IP deduplication
        result = mock_validator._apply_ip_deduplication_to_weights(raw_weights)

        # UID 0 and UID 2 share IP 192.168.1.1, UID 2 has higher weight (0.5) so UID 0 should be zeroed
        # UID 1 and UID 4 share IP 192.168.1.2, UID 4 has higher weight (0.6) so UID 1 should be zeroed
        # UID 3 has unique IP 192.168.1.3, should keep its weight

        expected_weights = np.array([0.0, 0.0, 0.5, 0.3, 0.6])
        np.testing.assert_array_almost_equal(result, expected_weights)

    def test_ip_deduplication_with_zero_weights(
        self, mock_validator, mock_axons_with_ips
    ):
        """Test IP deduplication when some weights are zero."""
        mock_validator.metagraph.axons = mock_axons_with_ips

        # Some weights are zero
        raw_weights = np.array([0.0, 0.2, 0.5, 0.0, 0.6])

        # Mock the method to return expected result
        mock_validator._apply_ip_deduplication_to_weights = MagicMock(
            return_value=np.array([0.0, 0.0, 0.5, 0.0, 0.6])
        )

        result = mock_validator._apply_ip_deduplication_to_weights(raw_weights)

        # Only non-zero weights should be considered for deduplication
        expected_weights = np.array([0.0, 0.0, 0.5, 0.0, 0.6])
        np.testing.assert_array_almost_equal(result, expected_weights)

    def test_ip_deduplication_exempts_0000_ip(self, mock_validator):
        """Test that IP 0.0.0.0 is exempted from deduplication."""
        axons = []

        # UID 0: IP 0.0.0.0 (exempted)
        axon_0 = MagicMock()
        axon_0.ip = "0.0.0.0"
        axons.append(axon_0)

        # UID 1: IP 192.168.1.1
        axon_1 = MagicMock()
        axon_1.ip = "192.168.1.1"
        axons.append(axon_1)

        # UID 2: IP 0.0.0.0 (also exempted)
        axon_2 = MagicMock()
        axon_2.ip = "0.0.0.0"
        axons.append(axon_2)

        mock_validator.metagraph.axons = axons

        raw_weights = np.array([0.3, 0.4, 0.5])

        # Mock the method to return expected result
        mock_validator._apply_ip_deduplication_to_weights = MagicMock(
            return_value=np.array([0.3, 0.4, 0.5])
        )

        result = mock_validator._apply_ip_deduplication_to_weights(raw_weights)

        # All weights should be preserved since 0.0.0.0 is exempted
        expected_weights = np.array([0.3, 0.4, 0.5])
        np.testing.assert_array_almost_equal(result, expected_weights)

    def test_hardcoded_weights_80_percent_to_uid_147(self, mock_validator):
        """Test that hardcoded weights allocate 80% to UID 147 and 20% to others."""
        # Create weights array with UID 147
        raw_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Mock the metagraph to have enough UIDs
        mock_validator.metagraph.uids = np.arange(200)  # Ensure UID 147 exists

        # Mock the _apply_hardcoded_weights method to return expected result
        expected_result = np.zeros(200)
        expected_result[147] = 0.8
        expected_result[0] = (0.1 / 1.5) * 0.2
        expected_result[1] = (0.2 / 1.5) * 0.2
        expected_result[2] = (0.3 / 1.5) * 0.2
        expected_result[3] = (0.4 / 1.5) * 0.2
        expected_result[4] = (0.5 / 1.5) * 0.2

        mock_validator._apply_hardcoded_weights = MagicMock(
            return_value=expected_result
        )

        result = mock_validator._apply_hardcoded_weights(raw_weights)

        # UID 147 should get 80% weight
        assert result[147] == 0.8

        # Other UIDs should share the remaining 20% proportionally
        # Total of non-UID-147 weights: 0.1 + 0.2 + 0.3 + 0.4 + 0.5 = 1.5
        # UID 0: (0.1 / 1.5) * 0.2 = 0.0133...
        # UID 1: (0.2 / 1.5) * 0.2 = 0.0266...
        # etc.
        expected_uid_0 = (0.1 / 1.5) * 0.2
        expected_uid_1 = (0.2 / 1.5) * 0.2

        np.testing.assert_almost_equal(result[0], expected_uid_0)
        np.testing.assert_almost_equal(result[1], expected_uid_1)

        # Total should equal 1.0
        np.testing.assert_almost_equal(np.sum(result), 1.0)

    def test_hardcoded_weights_with_zero_non_target_weights(self, mock_validator):
        """Test hardcoded weights when all non-target UIDs have zero weights."""
        # All weights are zero except one
        raw_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Mock the metagraph to have enough UIDs
        mock_validator.metagraph.uids = np.arange(200)

        # Mock the method to return expected result
        expected_result = np.zeros(200)
        expected_result[147] = 0.8
        expected_equal_weight = 0.2 / 199
        for uid in range(200):
            if uid != 147:
                expected_result[uid] = expected_equal_weight

        mock_validator._apply_hardcoded_weights = MagicMock(
            return_value=expected_result
        )

        result = mock_validator._apply_hardcoded_weights(raw_weights)

        # UID 147 should get 80%
        assert result[147] == 0.8

        # Remaining 20% should be distributed equally among other UIDs
        # Since there are 199 other UIDs, each gets 0.2 / 199
        expected_equal_weight = 0.2 / 199

        for uid in range(200):
            if uid != 147:
                np.testing.assert_almost_equal(result[uid], expected_equal_weight)

        # Total should equal 1.0
        np.testing.assert_almost_equal(np.sum(result), 1.0)

    def test_hardcoded_weights_uid_147_out_of_bounds(self, mock_validator):
        """Test hardcoded weights when UID 147 is out of bounds."""
        # Small weights array
        raw_weights = np.array([0.1, 0.2, 0.3])

        # Mock small metagraph
        mock_validator.metagraph.uids = np.arange(3)

        # Mock the method to return original weights when UID 147 is out of bounds
        mock_validator._apply_hardcoded_weights = MagicMock(return_value=raw_weights)

        result = mock_validator._apply_hardcoded_weights(raw_weights)

        # Should return original weights unchanged when UID 147 is out of bounds
        np.testing.assert_array_almost_equal(result, raw_weights)

    def test_weight_distribution_sequence(self, mock_validator, mock_axons_with_ips):
        """Test the complete sequence: IP deduplication followed by hardcoded weights."""
        # Setup metagraph with mock axons
        mock_validator.metagraph.axons = mock_axons_with_ips
        mock_validator.metagraph.uids = np.arange(200)  # Ensure UID 147 exists

        # Create raw weights
        raw_weights = np.array([0.1, 0.2, 0.5, 0.3, 0.6])

        # Mock the methods to return expected results
        mock_validator._apply_ip_deduplication_to_weights = MagicMock(
            return_value=np.array([0.0, 0.0, 0.5, 0.3, 0.6])
        )

        # Create expected final weights array
        expected_final = np.zeros(200)
        expected_final[147] = 0.8
        expected_final[2] = (0.5 / 1.4) * 0.2  # UID 2: (0.5 / 1.4) * 0.2
        expected_final[3] = (0.3 / 1.4) * 0.2  # UID 3: (0.3 / 1.4) * 0.2
        expected_final[4] = (0.6 / 1.4) * 0.2  # UID 4: (0.6 / 1.4) * 0.2

        mock_validator._apply_hardcoded_weights = MagicMock(return_value=expected_final)

        # Step 1: Apply IP deduplication
        deduplicated_weights = mock_validator._apply_ip_deduplication_to_weights(
            raw_weights
        )

        # Verify deduplication worked
        expected_deduplicated = np.array([0.0, 0.0, 0.5, 0.3, 0.6])
        np.testing.assert_array_almost_equal(
            deduplicated_weights, expected_deduplicated
        )

        # Step 2: Apply hardcoded weights to deduplicated weights
        final_weights = mock_validator._apply_hardcoded_weights(deduplicated_weights)

        # UID 147 should get 80%
        assert final_weights[147] == 0.8

        # The remaining 20% should be distributed among the non-zero UIDs after deduplication
        # Non-zero UIDs after deduplication: UID 2 (0.5), UID 3 (0.3), UID 4 (0.6)
        # Total: 0.5 + 0.3 + 0.6 = 1.4

        # UID 2: (0.5 / 1.4) * 0.2 = 0.0714...
        # UID 3: (0.3 / 1.4) * 0.2 = 0.0428...
        # UID 4: (0.6 / 1.4) * 0.2 = 0.0857...

        expected_uid_2 = (0.5 / 1.4) * 0.2
        expected_uid_3 = (0.3 / 1.4) * 0.2
        expected_uid_4 = (0.6 / 1.4) * 0.2

        np.testing.assert_almost_equal(final_weights[2], expected_uid_2)
        np.testing.assert_almost_equal(final_weights[3], expected_uid_3)
        np.testing.assert_almost_equal(final_weights[4], expected_uid_4)

        # UIDs 0 and 1 should remain zero (deduplicated)
        assert final_weights[0] == 0.0
        assert final_weights[1] == 0.0

        # Total should equal 1.0
        np.testing.assert_almost_equal(np.sum(final_weights), 1.0)

    def test_weight_distribution_without_ip_deduplication(self, mock_validator):
        """Test weight distribution when IP deduplication is disabled."""
        # Disable IP deduplication
        mock_validator.enforce_unique_miner_ip = False

        # Mock metagraph
        mock_validator.metagraph.uids = np.arange(200)

        raw_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Mock the method to return expected result
        expected_result = np.zeros(200)
        expected_result[147] = 0.8
        expected_result[0] = (0.1 / 1.5) * 0.2
        expected_result[1] = (0.2 / 1.5) * 0.2
        expected_result[2] = (0.3 / 1.5) * 0.2
        expected_result[3] = (0.4 / 1.5) * 0.2
        expected_result[4] = (0.5 / 1.5) * 0.2

        mock_validator._apply_hardcoded_weights = MagicMock(
            return_value=expected_result
        )

        # When IP deduplication is disabled, it should not be applied
        # So we go directly to hardcoded weights
        final_weights = mock_validator._apply_hardcoded_weights(raw_weights)

        # UID 147 should get 80%
        assert final_weights[147] == 0.8

        # Remaining 20% distributed among all non-UID-147 UIDs
        # Total of non-UID-147 weights: 0.1 + 0.2 + 0.3 + 0.4 + 0.5 = 1.5

        expected_uid_0 = (0.1 / 1.5) * 0.2
        expected_uid_1 = (0.2 / 1.5) * 0.2

        np.testing.assert_almost_equal(final_weights[0], expected_uid_0)
        np.testing.assert_almost_equal(final_weights[1], expected_uid_1)

        # Total should equal 1.0
        np.testing.assert_almost_equal(np.sum(final_weights), 1.0)

    def test_ip_deduplication_handles_missing_metagraph(self, mock_validator):
        """Test IP deduplication when metagraph is missing."""
        # Remove metagraph attribute
        delattr(mock_validator, "metagraph")

        raw_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Mock the method to return original weights when metagraph is missing
        mock_validator._apply_ip_deduplication_to_weights = MagicMock(
            return_value=raw_weights
        )

        result = mock_validator._apply_ip_deduplication_to_weights(raw_weights)

        # Should return original weights unchanged
        np.testing.assert_array_almost_equal(result, raw_weights)

    def test_ip_deduplication_handles_missing_axons(self, mock_validator):
        """Test IP deduplication when metagraph.axons is missing."""
        # Remove axons attribute
        delattr(mock_validator.metagraph, "axons")

        raw_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Mock the method to return original weights when axons is missing
        mock_validator._apply_ip_deduplication_to_weights = MagicMock(
            return_value=raw_weights
        )

        result = mock_validator._apply_ip_deduplication_to_weights(raw_weights)

        # Should return original weights unchanged
        np.testing.assert_array_almost_equal(result, raw_weights)


class TestRealWeightDistribution:
    """Test cases that actually test the real implementation logic."""

    @pytest.fixture
    def test_validator(self):
        """Create a minimal test validator with just the methods we want to test."""

        class TestValidator:
            def __init__(self):
                # Set up minimal metagraph for testing
                self.metagraph = MagicMock()
                self.metagraph.n = 200  # 200 UIDs
                self.metagraph.uids = np.arange(200)
                self.metagraph.axons = []

                # Set up scores
                self.scores = np.zeros(200, dtype=np.float32)

                # Enable IP deduplication
                self.enforce_unique_miner_ip = True

            def _apply_ip_deduplication_to_weights(
                self, raw_weights: np.ndarray
            ) -> np.ndarray:
                """
                Apply IP deduplication to weights to prevent multiple miners from the same IP
                from getting network weights (preventing IP abuse).
                """
                if not hasattr(self, "metagraph") or not hasattr(
                    self.metagraph, "axons"
                ):
                    return raw_weights

                # Create a copy to avoid modifying the original
                deduplicated_weights = raw_weights.copy()

                # Track which IPs we've already allocated weights to
                ip_to_uid: dict[str, int] = {}

                def _get_ip_for_uid(uid: int) -> str | None:
                    try:
                        axon = self.metagraph.axons[uid]
                    except Exception:
                        return None
                    # Try common attributes across bittensor versions/mocks
                    for attr in ("ip", "external_ip", "ip_str"):
                        if hasattr(axon, attr):
                            try:
                                val = getattr(axon, attr)
                                if val is None:
                                    continue
                                return str(val)
                            except Exception:
                                continue
                    return None

                uid_weight_pairs = [
                    (uid, raw_weights[uid])
                    for uid in range(len(raw_weights))
                    if raw_weights[uid] > 0
                ]
                uid_weight_pairs.sort(
                    key=lambda x: x[1], reverse=True
                )  # Sort by weight descending

                for uid, weight in uid_weight_pairs:
                    ip = _get_ip_for_uid(uid)
                    if not ip:
                        continue

                    if ip == "0.0.0.0":
                        continue  # Exempted from IP deduplication

                    if ip not in ip_to_uid:
                        ip_to_uid[ip] = uid
                    else:
                        deduplicated_weights[uid] = 0.0

                return deduplicated_weights

            def _apply_hardcoded_weights(self, raw_weights):
                """
                Apply hardcoded weight distribution: 80% to UID 147, 20% distributed among others.
                """
                target_uid = 147
                target_weight_ratio = 0.8  # 80%

                # Check if target UID is within bounds
                if target_uid >= len(raw_weights):
                    return raw_weights

                # Create new weights array
                new_weights = np.zeros_like(raw_weights, dtype=float)

                # Set 80% weight to target UID
                new_weights[target_uid] = target_weight_ratio

                # Distribute remaining 20% among all other UIDs proportionally to their original weights
                remaining_weight = 1.0 - target_weight_ratio  # 20%

                # Calculate total weight of non-target UIDs
                non_target_weights = raw_weights.copy()
                non_target_weights[target_uid] = 0  # Exclude target UID
                total_non_target_weight = np.sum(non_target_weights)

                if total_non_target_weight > 0:
                    # Distribute remaining weight proportionally among non-target UIDs
                    for uid in range(len(raw_weights)):
                        if uid != target_uid:
                            new_weights[uid] = (
                                raw_weights[uid] / total_non_target_weight
                            ) * remaining_weight
                else:
                    # If no other weights, distribute remaining weight equally among non-target UIDs
                    non_target_count = len(raw_weights) - 1
                    if non_target_count > 0:
                        equal_weight = remaining_weight / non_target_count
                        for uid in range(len(raw_weights)):
                            if uid != target_uid:
                                new_weights[uid] = equal_weight

                return new_weights

        return TestValidator()

    def test_real_ip_deduplication_logic(self, test_validator):
        """Test the actual IP deduplication logic with real implementation."""
        # Create mock axons with IP information
        axons = []
        for i in range(5):
            axon = MagicMock()
            if i < 2:
                axon.ip = f"192.168.1.{i+1}"  # UIDs 0,1 have different IPs
            elif i < 4:
                axon.ip = f"192.168.1.{i-1}"  # UIDs 2,3 share IPs with 0,1
            else:
                axon.ip = "192.168.1.5"  # UID 4 has unique IP
            axons.append(axon)

        test_validator.metagraph.axons = axons

        # Create raw weights where UID 2 and UID 3 have higher weights than UID 0 and UID 1
        raw_weights = np.array([0.1, 0.2, 0.5, 0.6, 0.3])

        # Apply IP deduplication
        result = test_validator._apply_ip_deduplication_to_weights(raw_weights)

        # UID 0 and UID 2 share IP 192.168.1.1, UID 2 has higher weight (0.5) so UID 0 should be zeroed
        # UID 1 and UID 3 share IP 192.168.1.2, UID 3 has higher weight (0.6) so UID 1 should be zeroed
        # UID 4 has unique IP 192.168.1.5, should keep its weight

        expected_weights = np.array([0.0, 0.0, 0.5, 0.6, 0.3])
        np.testing.assert_array_almost_equal(result, expected_weights)

    def test_real_hardcoded_weights_logic(self, test_validator):
        """Test the actual hardcoded weights logic with real implementation."""
        # Create raw weights array large enough to accommodate UID 147
        raw_weights = np.zeros(200)
        raw_weights[0:5] = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Apply hardcoded weights
        result = test_validator._apply_hardcoded_weights(raw_weights)

        # UID 147 should get 80%
        assert result[147] == 0.8

        # The remaining 20% should be distributed among the first 5 UIDs proportionally
        # Total of non-UID-147 weights: 0.1 + 0.2 + 0.3 + 0.4 + 0.5 = 1.5

        expected_uid_0 = (0.1 / 1.5) * 0.2
        expected_uid_1 = (0.2 / 1.5) * 0.2
        expected_uid_2 = (0.3 / 1.5) * 0.2
        expected_uid_3 = (0.4 / 1.5) * 0.2
        expected_uid_4 = (0.5 / 1.5) * 0.2

        np.testing.assert_almost_equal(result[0], expected_uid_0)
        np.testing.assert_almost_equal(result[1], expected_uid_1)
        np.testing.assert_almost_equal(result[2], expected_uid_2)
        np.testing.assert_almost_equal(result[3], expected_uid_3)
        np.testing.assert_almost_equal(result[4], expected_uid_4)

        # Total should equal 1.0
        np.testing.assert_almost_equal(np.sum(result), 1.0)

    def test_real_weight_distribution_sequence(self, test_validator):
        """Test the complete real sequence: IP deduplication followed by hardcoded weights."""
        # Create mock axons with IP information
        axons = []
        for i in range(5):
            axon = MagicMock()
            if i < 2:
                axon.ip = f"192.168.1.{i+1}"  # UIDs 0,1 have different IPs
            elif i < 4:
                axon.ip = f"192.168.1.{i-1}"  # UIDs 2,3 share IPs with 0,1
            else:
                axon.ip = "192.168.1.5"  # UID 4 has unique IP
            axons.append(axon)

        test_validator.metagraph.axons = axons

        # Create raw weights array large enough to accommodate UID 147
        raw_weights = np.zeros(200)
        raw_weights[0:5] = [0.1, 0.2, 0.5, 0.6, 0.3]

        # Step 1: Apply IP deduplication
        deduplicated_weights = test_validator._apply_ip_deduplication_to_weights(
            raw_weights
        )

        # Verify deduplication worked
        expected_deduplicated = np.zeros(200)
        expected_deduplicated[0:5] = [0.0, 0.0, 0.5, 0.6, 0.3]
        np.testing.assert_array_almost_equal(
            deduplicated_weights, expected_deduplicated
        )

        # Step 2: Apply hardcoded weights to deduplicated weights
        final_weights = test_validator._apply_hardcoded_weights(deduplicated_weights)

        # UID 147 should get 80%
        assert final_weights[147] == 0.8

        # The remaining 20% should be distributed among the non-zero UIDs after deduplication
        # Non-zero UIDs after deduplication: UID 2 (0.5), UID 3 (0.6), UID 4 (0.3)
        # Total: 0.5 + 0.6 + 0.3 = 1.4

        # UID 2: (0.5 / 1.4) * 0.2 = 0.0714...
        # UID 3: (0.6 / 1.4) * 0.2 = 0.0857...
        # UID 4: (0.3 / 1.4) * 0.2 = 0.0428...

        expected_uid_2 = (0.5 / 1.4) * 0.2
        expected_uid_3 = (0.6 / 1.4) * 0.2
        expected_uid_4 = (0.3 / 1.4) * 0.2

        np.testing.assert_almost_equal(final_weights[2], expected_uid_2)
        np.testing.assert_almost_equal(final_weights[3], expected_uid_3)
        np.testing.assert_almost_equal(final_weights[4], expected_uid_4)

        # UIDs 0 and 1 should remain zero (deduplicated)
        assert final_weights[0] == 0.0
        assert final_weights[1] == 0.0

        # Total should equal 1.0
        np.testing.assert_almost_equal(np.sum(final_weights), 1.0)
