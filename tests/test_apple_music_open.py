"""Music.app opening boundary: validate Apple identity and fail closed before launch."""

import subprocess
import unittest
from unittest.mock import patch

from music_agent.apple_music_open import (
    AppleMusicOpenError,
    AppleMusicOpenUnavailableError,
    client_song_url,
    open_music_app,
)


class OpenMusicAppTest(unittest.TestCase):
    TRACK_ID = "1258917044"

    def test_real_apple_urls_preserve_storefront_and_target_the_concrete_song(self) -> None:
        cases = (
            (
                "https://music.apple.com/us/album/reputation/1258917041?i=1258917044",
                "https://music.apple.com/us/song/1258917044",
            ),
            (
                "https://itunes.apple.com/cn/album/id1258917041?i=1258917044",
                "https://music.apple.com/cn/song/1258917044",
            ),
            (
                "https://geo.music.apple.com/gb/album/x?i=1258917044",
                "https://music.apple.com/gb/song/1258917044",
            ),
        )
        for track_view_url, expected in cases:
            with self.subTest(track_view_url=track_view_url):
                self.assertEqual(client_song_url(track_view_url, self.TRACK_ID), expected)

    def test_music_app_handoff_uses_argument_vector_not_a_shell(self) -> None:
        track_view_url = (
            "https://music.apple.com/us/album/reputation/1258917041?i=1258917044"
        )
        with patch("subprocess.run") as mocked_run:
            client_url = open_music_app(track_view_url, self.TRACK_ID)
        self.assertEqual(client_url, "https://music.apple.com/us/song/1258917044")
        mocked_run.assert_called_once_with(
            ["/usr/bin/open", "-a", "Music", client_url],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
        )

    def test_foreign_hosts_are_refused_before_any_side_effect(self) -> None:
        for url in (
            "https://evil.example.com/us/album/x",
            "https://music.apple.com.evil.example.com/us/album/x",
            "https://musicapple.com/us/album/x",
        ):
            with self.subTest(url=url):
                with patch("subprocess.run") as mocked_run:
                    with self.assertRaises(AppleMusicOpenError):
                        open_music_app(url, self.TRACK_ID)
                mocked_run.assert_not_called()

    def test_non_http_missing_storefront_and_invalid_track_ids_are_refused(self) -> None:
        cases = (
            ("music://music.apple.com/us/album/x", self.TRACK_ID),
            ("file:///etc/passwd", self.TRACK_ID),
            ("https://music.apple.com/album/x", self.TRACK_ID),
            ("https://music.apple.com/usa/album/x", self.TRACK_ID),
            ("https://music.apple.com/us/album/x", "12;open -a Calculator"),
            ("https://music.apple.com/us/album/x", ""),
        )
        for url, track_id in cases:
            with self.subTest(url=url, track_id=track_id):
                with patch("subprocess.run") as mocked_run:
                    with self.assertRaises(AppleMusicOpenError):
                        open_music_app(url, track_id)
                mocked_run.assert_not_called()

    def test_launchservices_failure_is_a_typed_failure(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["/usr/bin/open"]),
        ):
            with self.assertRaises(AppleMusicOpenError):
                open_music_app(
                    "https://music.apple.com/us/album/x", self.TRACK_ID
                )

    def test_unavailable_error_carries_its_own_code(self) -> None:
        error = AppleMusicOpenUnavailableError("no link")
        self.assertEqual(error.code, "apple_music_open_unavailable")
        self.assertEqual(AppleMusicOpenError("x").code, "apple_music_open_failed")


if __name__ == "__main__":
    unittest.main()
