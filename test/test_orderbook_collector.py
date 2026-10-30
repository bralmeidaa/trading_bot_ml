"""
Tests for the order book collector's pure feature computation (no network).
"""
import pytest

from backend.data.orderbook_collector import compute_features, OrderBookCollector


def _book(bids, asks):
    return {"bids": bids, "asks": asks}


class TestComputeFeatures:
    def test_basic_mid_and_spread(self):
        ob = _book([[100.0, 1.0]], [[100.1, 1.0]])
        f = compute_features(ob, "BTC/USDT", 123)
        assert f["mid"] == pytest.approx(100.05)
        assert f["spread_bps"] == pytest.approx((0.1 / 100.05) * 1e4, rel=1e-6)
        assert f["symbol"] == "BTC/USDT" and f["timestamp"] == 123

    def test_imbalance_positive_when_more_bids(self):
        ob = _book([[100, 10], [99, 10]], [[101, 1], [102, 1]])
        f = compute_features(ob, "X", 0)
        assert f["imbalance_top5"] > 0          # bid-heavy → positive imbalance

    def test_imbalance_negative_when_more_asks(self):
        ob = _book([[100, 1], [99, 1]], [[101, 10], [102, 10]])
        f = compute_features(ob, "X", 0)
        assert f["imbalance_top5"] < 0

    def test_imbalance_bounded(self):
        ob = _book([[100, 5]], [[101, 3]])
        f = compute_features(ob, "X", 0)
        assert -1.0 <= f["imbalance_top20"] <= 1.0

    def test_microprice_between_bid_ask(self):
        ob = _book([[100.0, 2.0]], [[101.0, 8.0]])
        f = compute_features(ob, "X", 0)
        assert 100.0 <= f["microprice"] <= 101.0

    def test_empty_book_returns_none(self):
        assert compute_features(_book([], []), "X", 0) is None
        assert compute_features(_book([[100, 1]], []), "X", 0) is None

    def test_crossed_book_returns_none(self):
        # ask below bid = degenerate
        assert compute_features(_book([[101, 1]], [[100, 1]]), "X", 0) is None


class TestStorage:
    def test_append_writes_csv(self, tmp_path):
        col = OrderBookCollector(["X"], storage_dir=str(tmp_path))
        feat = compute_features(_book([[100, 1]], [[100.1, 1]]), "X", 1)
        col.append(feat)
        col.append(feat)
        files = list(tmp_path.glob("orderbook_*.csv"))
        assert len(files) == 1
        content = files[0].read_text().strip().splitlines()
        assert content[0].startswith("timestamp,symbol")   # header once
        assert len(content) == 3                            # header + 2 rows

    def test_status_shape(self, tmp_path):
        col = OrderBookCollector(["X", "Y"], storage_dir=str(tmp_path), interval_sec=30)
        s = col.status()
        assert s["symbols"] == 2 and s["interval_sec"] == 30
        assert "today_file" in s and "snapshots_written" in s
