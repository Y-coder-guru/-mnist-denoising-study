"""Offline downloader regression checks; no network or MNIST downloads."""

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mnist_download import fetch_archive, make_opener


class Response(io.BytesIO):
    def __init__(self, payload, advertised_size=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if advertised_size is None else advertised_size)}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        test_directory = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(dir=test_directory)
        self.raw = Path(self.temporary.name)
        if self.raw.resolve().parent != test_directory:
            raise RuntimeError("Test temporary directory escaped the test workspace")
        self.addCleanup(self.temporary.cleanup)
        self.payload = b"verified archive payload"
        self.expected = hashlib.md5(self.payload, usedforsecurity=False).hexdigest()
        self.mirrors = ("https://first.example/", "https://second.example/")

    def fetch(self, opener, mirrors=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return fetch_archive(self.raw, "test.gz", self.expected, opener=opener,
                                 retries=1, mirrors=self.mirrors if mirrors is None else mirrors)

    def test_verified_cache_does_not_contact_network(self):
        (self.raw / "test.gz").write_bytes(self.payload)
        opener = Mock()
        self.fetch(opener)
        opener.open.assert_not_called()

    def test_checksum_failure_switches_to_next_mirror(self):
        opener = Mock()
        opener.open.side_effect = [Response(b"wrong payload"), Response(self.payload)]
        result = self.fetch(opener)
        self.assertEqual((self.raw / "test.gz").read_bytes(), self.payload)
        self.assertEqual(result["source"], self.mirrors[1] + "test.gz")
        self.assertEqual(opener.open.call_count, 2)

    def test_truncated_response_is_rejected_then_retried(self):
        opener = Mock()
        opener.open.side_effect = [Response(self.payload[:3], len(self.payload)), Response(self.payload)]
        self.fetch(opener)
        self.assertEqual((self.raw / "test.gz").read_bytes(), self.payload)
        self.assertFalse((self.raw / "test.gz.part").exists())

    def test_failed_download_never_replaces_existing_destination(self):
        original = b"old incomplete archive"
        (self.raw / "test.gz").write_bytes(original)
        opener = Mock()
        opener.open.side_effect = lambda *args, **kwargs: Response(b"invalid")
        with self.assertRaises(RuntimeError):
            self.fetch(opener)
        self.assertEqual((self.raw / "test.gz").read_bytes(), original)

    def test_direct_mode_explicitly_disables_proxy_handler(self):
        with patch("mnist_download.urllib.request.build_opener") as build:
            make_opener(no_proxy=True)
        handler = build.call_args.args[0]
        self.assertEqual(handler.proxies, {})


if __name__ == "__main__":
    unittest.main()
