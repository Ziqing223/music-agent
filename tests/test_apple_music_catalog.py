"""P11.1: Apple Music Catalog search adapter tests (transport, parsing, credentials)."""

import unittest
from unittest.mock import patch

from music_agent.apple_music_catalog import (
    DEFAULT_DEVELOPER_TOKEN_ENV,
    AppleMusicCatalogAdapter,
    CatalogCredentialsError,
    CatalogMappingError,
    CatalogTransportError,
    CatalogTrack,
    MusicKitTransport,
)


MUSICKIT_SEARCH_PAYLOAD = {
    "results": {
        "songs": {
            "data": [
                {
                    "id": "1437682044",
                    "type": "songs",
                    "attributes": {
                        "name": "Catalog Anthem",
                        "artistName": "Artist Alpha",
                        "albumName": "The Catalog Album",
                        "genreNames": ["Electronic", "Dance"],
                        "durationInMillis": 201000,
                        "releaseDate": "2024-01-15",
                        "isrc": "USSYN2400001",
                        "url": "https://music.apple.com/us/album/catalog-anthem/1437682044?i=1437682044",
                    },
                },
                {
                    "id": "1437682045",
                    "type": "songs",
                    "attributes": {
                        "name": "No-ISRC Song",
                        "artistName": "Artist Beta",
                        "albumName": "Singles",
                        "genreNames": ["Rock"],
                    },
                },
            ]
        }
    }
}


class FakeTransport:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def search(self, term: str, limit: int) -> str:
        self.calls.append((term, limit))
        return self.response


class CatalogParsingTest(unittest.TestCase):
    def test_valid_musickit_payload_maps_to_catalog_tracks(self) -> None:
        adapter = AppleMusicCatalogAdapter(FakeTransport(_json(MUSICKIT_SEARCH_PAYLOAD)))
        tracks = adapter.parse_search_results(_json(MUSICKIT_SEARCH_PAYLOAD))
        self.assertEqual(len(tracks), 2)
        first = tracks[0]
        self.assertIsInstance(first, CatalogTrack)
        self.assertEqual(first.catalog_id, "1437682044")
        self.assertEqual(first.name, "Catalog Anthem")
        self.assertEqual(first.artist_names, ("Artist Alpha",))
        self.assertEqual(first.album_name, "The Catalog Album")
        self.assertEqual(first.genres, ("Electronic", "Dance"))
        self.assertEqual(first.isrc, "USSYN2400001")
        self.assertEqual(first.duration_ms, 201000)
        self.assertEqual(first.release_date, "2024-01-15")
        self.assertIsNotNone(first.url)
        second = tracks[1]
        self.assertIsNone(second.isrc)
        self.assertIsNone(second.duration_ms)
        self.assertIsNone(second.release_date)

    def test_search_delegates_to_transport_with_term_and_limit(self) -> None:
        transport = FakeTransport(_json(MUSICKIT_SEARCH_PAYLOAD))
        adapter = AppleMusicCatalogAdapter(transport)
        tracks = adapter.search("catalog anthem", 5)
        self.assertEqual(transport.calls, [("catalog anthem", 5)])
        self.assertEqual(len(tracks), 2)

    def test_empty_result_is_valid(self) -> None:
        transport = FakeTransport('{"results": {"songs": {"data": []}}}')
        self.assertEqual(AppleMusicCatalogAdapter(transport).search("nothing", 1), ())

    def test_non_json_response_fails_closed(self) -> None:
        adapter = AppleMusicCatalogAdapter(FakeTransport("not json"))
        with self.assertRaises(CatalogMappingError):
            adapter.parse_search_results("not json")

    def test_missing_results_songs_data_fails_closed(self) -> None:
        adapter = AppleMusicCatalogAdapter(FakeTransport("{}"))
        with self.assertRaises(CatalogMappingError):
            adapter.parse_search_results('{"results": {}}')

    def test_song_without_id_fails_closed(self) -> None:
        payload = {"results": {"songs": {"data": [{"attributes": {"name": "x", "artistName": "y"}}]}}}
        adapter = AppleMusicCatalogAdapter(FakeTransport("{}"))
        with self.assertRaises(CatalogMappingError):
            adapter.parse_search_results(_json(payload))

    def test_song_without_name_or_artist_fails_closed(self) -> None:
        payload = {"results": {"songs": {"data": [{"id": "1", "attributes": {"name": "x"}}]}}}
        adapter = AppleMusicCatalogAdapter(FakeTransport("{}"))
        with self.assertRaises(CatalogMappingError):
            adapter.parse_search_results(_json(payload))

    def test_relationship_ids_map_to_authoritative_artist_and_album_identity(self) -> None:
        payload = {"results": {"songs": {"data": [
            {
                "id": "1437682044",
                "type": "songs",
                "attributes": {"name": "x", "artistName": "A", "albumName": "B"},
                "relationships": {
                    "artists": {"data": [
                        {"id": "ART-1", "type": "artists"},
                        {"id": "ART-2", "type": "artists"},
                    ]},
                    "albums": {"data": [{"id": "ALB-1", "type": "albums"}]},
                },
            },
        ]}}}
        adapter = AppleMusicCatalogAdapter(FakeTransport("{}"))
        track = adapter.parse_search_results(_json(payload))[0]
        self.assertEqual(track.artist_catalog_ids, ("ART-1", "ART-2"))
        self.assertEqual(track.album_catalog_id, "ALB-1")

    def test_missing_relationships_yield_no_identity_evidence(self) -> None:
        adapter = AppleMusicCatalogAdapter(FakeTransport("{}"))
        track = adapter.parse_search_results(_json(MUSICKIT_SEARCH_PAYLOAD))[0]
        self.assertEqual(track.artist_catalog_ids, ())
        self.assertIsNone(track.album_catalog_id)

    def test_malformed_relationship_data_fails_closed(self) -> None:
        payload = {"results": {"songs": {"data": [
            {
                "id": "1",
                "type": "songs",
                "attributes": {"name": "x", "artistName": "A"},
                "relationships": {"artists": {"data": [{"type": "artists"}]}},
            },
        ]}}}
        adapter = AppleMusicCatalogAdapter(FakeTransport("{}"))
        with self.assertRaises(CatalogMappingError):
            adapter.parse_search_results(_json(payload))


class MusicKitTransportTest(unittest.TestCase):
    def test_missing_developer_token_raises_actionable_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(CatalogCredentialsError) as context:
                MusicKitTransport().search("x", 1)
        self.assertIn(DEFAULT_DEVELOPER_TOKEN_ENV, str(context.exception))

    def test_request_carries_bearer_token_and_quoted_query(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def read(self) -> bytes:
                return b'{"results": {"songs": {"data": []}}}'

        with patch("music_agent.apple_music_catalog.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse()
            transport = MusicKitTransport(developer_token="dev-token", user_token="user-token")
            text = transport.search("hello world", 3)
            self.assertEqual(text, '{"results": {"songs": {"data": []}}}')
            request = urlopen.call_args.args[0]
            self.assertEqual(request.headers["Authorization"], "Bearer dev-token")
            self.assertEqual(request.headers["Music-user-token"], "user-token")
            self.assertIn("/v1/catalog/us/search", request.full_url)
            self.assertIn("term=hello%20world", request.full_url)
            self.assertIn("limit=3", request.full_url)

    def test_http_error_raises_transport_error(self) -> None:
        import urllib.error

        with patch("music_agent.apple_music_catalog.urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.HTTPError(
                "https://api.music.apple.com/", 401, "Unauthorized", {}, None
            )
            transport = MusicKitTransport(developer_token="dev-token")
            with self.assertRaises(CatalogTransportError) as context:
                transport.search("x", 1)
        self.assertIn("401", str(context.exception))

    def test_invalid_arguments_fail_closed(self) -> None:
        transport = MusicKitTransport(developer_token="dev-token")
        with self.assertRaises(CatalogMappingError):
            transport.search("", 1)
        with self.assertRaises(CatalogMappingError):
            transport.search("x", 0)


def _json(payload: object) -> str:
    import json

    return json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
