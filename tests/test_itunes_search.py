"""P11-T3: credential-free iTunes Search adapter -- transport and parser.

Deterministic fake-transport tests for the iTunes Search API adapter: realistic payload
mapping under the ``itunes_store`` identity namespace (never inferred equal to
``apple_music_catalog``), fail-closed behavior on malformed payloads, and the
credential-free transport (no environment read, no Authorization header on the wire).
"""

import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from music_agent.apple_music_catalog import (
    AppleMusicCatalogAdapter,
    CatalogMappingError,
    CatalogTransportError,
)
from music_agent.catalog_ingestion import default_catalog_search_source
from music_agent.itunes_search import (
    ITUNES_SOURCE_SYSTEM,
    iTunesSearchAdapter,
    iTunesSearchTransport,
)


REALISTIC_RESULT = {
    "wrapperType": "track",
    "kind": "song",
    "artistId": 148607010,
    "collectionId": 1258917041,
    "trackId": 1258917044,
    "artistName": "Taylor Swift",
    "collectionName": "reputation",
    "trackName": "Delicate",
    "trackViewUrl": "https://music.apple.com/us/album/delicate/1258917041?i=1258917044&uo=4",
    "previewUrl": "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/delicate.m4a",
    "releaseDate": "2017-11-10T08:00:00Z",
    "trackTimeMillis": 232861,
    "primaryGenreName": "Pop",
    "isStreamable": True,
}


class FakeTransport:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[str, int]] = []
        self.lookup_calls: list[str] = []

    def search(self, term: str, limit: int) -> str:
        self.calls.append((term, limit))
        if self.error is not None:
            raise self.error
        return self.text

    def lookup(self, itunes_id: str) -> str:
        self.lookup_calls.append(itunes_id)
        if self.error is not None:
            raise self.error
        return self.text


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def read(self) -> bytes:
        return self._body


def payload(results: list) -> str:
    return json.dumps({"resultCount": len(results), "results": results})


class iTunesSearchAdapterTest(unittest.TestCase):
    def test_realistic_payload_maps_under_itunes_store_identity(self) -> None:
        track = iTunesSearchAdapter(FakeTransport(payload([REALISTIC_RESULT]))).search("delicate")[0]
        self.assertEqual(track.catalog_id, "1258917044")
        self.assertEqual(track.source_system, ITUNES_SOURCE_SYSTEM)
        self.assertNotEqual(track.source_system, "apple_music_catalog")
        self.assertEqual(track.name, "Delicate")
        self.assertEqual(track.artist_names, ("Taylor Swift",))
        self.assertEqual(track.album_name, "reputation")
        self.assertEqual(track.genres, ("Pop",))
        self.assertIsNone(track.isrc)  # iTunes supplies no ISRC; never fabricated
        self.assertEqual(track.duration_ms, 232861)
        self.assertEqual(track.release_date, "2017-11-10")
        self.assertEqual(track.url, REALISTIC_RESULT["trackViewUrl"])
        self.assertEqual(track.preview_url, REALISTIC_RESULT["previewUrl"])
        self.assertEqual(track.artist_catalog_ids, ("148607010",))
        self.assertEqual(track.album_catalog_id, "1258917041")

    def test_missing_identity_fields_fail_closed(self) -> None:
        for key in ("trackId", "artistId", "trackName", "artistName"):
            entry = {**REALISTIC_RESULT}
            del entry[key]
            with self.assertRaises(CatalogMappingError, msg=key):
                iTunesSearchAdapter(FakeTransport(payload([entry]))).search("x")

    def test_non_positive_numeric_ids_fail_closed(self) -> None:
        for key, value in (("trackId", 0), ("artistId", -1), ("collectionId", True)):
            entry = {**REALISTIC_RESULT}
            entry[key] = value
            with self.assertRaises(CatalogMappingError, msg=key):
                iTunesSearchAdapter(FakeTransport(payload([entry]))).search("x")

    def test_malformed_payloads_fail_closed(self) -> None:
        cases = ("", "[]", "not json", '{"resultCount": 1}', '{"results": {}}')
        for text in cases:
            with self.assertRaises(CatalogMappingError, msg=text):
                iTunesSearchAdapter(FakeTransport(text)).search("x")

    def test_empty_results_maps_to_no_tracks(self) -> None:
        self.assertEqual(iTunesSearchAdapter(FakeTransport(payload([]))).search("x"), ())

    def test_album_identity_may_be_absent(self) -> None:
        entry = {**REALISTIC_RESULT}
        del entry["collectionId"], entry["collectionName"]
        track = iTunesSearchAdapter(FakeTransport(payload([entry]))).search("x")[0]
        self.assertIsNone(track.album_catalog_id)
        self.assertIsNone(track.album_name)

    def test_unparseable_release_date_degrades_to_missing(self) -> None:
        entry = {**REALISTIC_RESULT}
        entry["releaseDate"] = "not-a-date"
        track = iTunesSearchAdapter(FakeTransport(payload([entry]))).search("x")[0]
        self.assertIsNone(track.release_date)

    def test_search_passes_term_and_limit_to_transport(self) -> None:
        transport = FakeTransport(payload([]))
        iTunesSearchAdapter(transport).search("起风了", 3)
        self.assertEqual(transport.calls, [("起风了", 3)])

    def test_adapter_rejects_bad_transport(self) -> None:
        with self.assertRaises(CatalogMappingError):
            iTunesSearchAdapter(object())  # type: ignore[arg-type]

    def test_lookup_resolves_preview_url_by_exact_track_id(self) -> None:
        transport = FakeTransport(payload([REALISTIC_RESULT]))
        self.assertEqual(
            iTunesSearchAdapter(transport).lookup_preview_url("1258917044"),
            REALISTIC_RESULT["previewUrl"],
        )
        self.assertEqual(transport.lookup_calls, ["1258917044"])

    def test_lookup_requires_exact_track_id_match(self) -> None:
        self.assertIsNone(
            iTunesSearchAdapter(FakeTransport(payload([REALISTIC_RESULT]))).lookup_preview_url("9999999999")
        )

    def test_lookup_without_preview_url_is_absent(self) -> None:
        entry = {**REALISTIC_RESULT}
        del entry["previewUrl"]
        self.assertIsNone(
            iTunesSearchAdapter(FakeTransport(payload([entry]))).lookup_preview_url("1258917044")
        )

    def test_lookup_malformed_payloads_fail_closed(self) -> None:
        for text in ("", "not json", "[]", '{"results": {}}'):
            with self.assertRaises(CatalogMappingError, msg=text):
                iTunesSearchAdapter(FakeTransport(text)).lookup_preview_url("1258917044")

    def test_lookup_empty_results_is_absent(self) -> None:
        self.assertIsNone(
            iTunesSearchAdapter(FakeTransport(payload([]))).lookup_preview_url("1258917044")
        )

    def test_lookup_resolves_track_view_url_by_exact_track_id(self) -> None:
        """P16-S4: the official track page URL resolves from the durable itunes_store
        binding with the same strict-trackId contract as the preview lookup."""
        transport = FakeTransport(payload([REALISTIC_RESULT]))
        self.assertEqual(
            iTunesSearchAdapter(transport).lookup_track_view_url("1258917044"),
            REALISTIC_RESULT["trackViewUrl"],
        )
        self.assertEqual(transport.lookup_calls, ["1258917044"])

    def test_track_view_lookup_requires_exact_track_id_match(self) -> None:
        self.assertIsNone(
            iTunesSearchAdapter(FakeTransport(payload([REALISTIC_RESULT]))).lookup_track_view_url(
                "9999999999"
            )
        )

    def test_track_view_lookup_without_url_is_absent(self) -> None:
        entry = {**REALISTIC_RESULT}
        del entry["trackViewUrl"]
        self.assertIsNone(
            iTunesSearchAdapter(FakeTransport(payload([entry]))).lookup_track_view_url(
                "1258917044"
            )
        )

    def test_track_view_lookup_malformed_payloads_fail_closed(self) -> None:
        for text in ("", "not json", "[]", '{"results": {}}'):
            with self.assertRaises(CatalogMappingError, msg=text):
                iTunesSearchAdapter(FakeTransport(text)).lookup_track_view_url("1258917044")

    def test_track_view_lookup_empty_results_is_absent(self) -> None:
        self.assertIsNone(
            iTunesSearchAdapter(FakeTransport(payload([]))).lookup_track_view_url("1258917044")
        )


class iTunesSearchTransportTest(unittest.TestCase):
    def test_requires_no_credentials_and_sends_no_authorization(self) -> None:
        captured: dict[str, dict] = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["headers"] = dict(request.headers.items())
            return FakeResponse(b'{"resultCount":0,"results":[]}')

        with patch.dict(os.environ, {}, clear=True):  # empty environment: nothing to read
            with patch("urllib.request.urlopen", fake_urlopen):
                text = iTunesSearchTransport().search("delicate", 1)
        self.assertIn('"results":[]', text)
        self.assertNotIn("Authorization", captured["headers"])

    def test_http_error_maps_to_typed_transport_error(self) -> None:
        error = HTTPError("https://itunes.apple.com/search", 500, "boom", None, BytesIO())

        def raise_http_error(request, timeout):  # type: ignore[no-untyped-def]
            raise error

        with patch("urllib.request.urlopen", raise_http_error):
            with self.assertRaises(CatalogTransportError) as raised:
                iTunesSearchTransport().search("x", 1)
        self.assertIn("HTTP 500", str(raised.exception))

    def test_url_error_maps_to_typed_transport_error(self) -> None:
        def raise_url_error(request, timeout):  # type: ignore[no-untyped-def]
            raise URLError("connection refused")

        with patch("urllib.request.urlopen", raise_url_error):
            with self.assertRaises(CatalogTransportError):
                iTunesSearchTransport().search("x", 1)

    def test_term_and_limit_validation(self) -> None:
        transport = iTunesSearchTransport()
        for term in ("", "   ", None):  # type: ignore[arg-type]
            with self.assertRaises(CatalogMappingError):
                transport.search(term, 1)  # type: ignore[arg-type]
        for limit in (0, -1, True, "25"):
            with self.assertRaises(CatalogMappingError):
                transport.search("x", limit)  # type: ignore[arg-type]

    def test_lookup_hits_lookup_endpoint_without_credentials(self) -> None:
        captured: dict[str, str] = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["url"] = request.full_url
            captured["auth"] = request.headers.get("Authorization", "")
            return FakeResponse(b'{"resultCount":1,"results":[]}')

        with patch("urllib.request.urlopen", fake_urlopen):
            text = iTunesSearchTransport().lookup("1258917044")
        self.assertIn('"results":[]', text)
        self.assertEqual(captured["url"], "https://itunes.apple.com/lookup?id=1258917044&country=us")
        self.assertEqual(captured["auth"], "")

    def test_lookup_id_validation(self) -> None:
        for itunes_id in ("", "   ", None):  # type: ignore[arg-type]
            with self.assertRaises(CatalogMappingError):
                iTunesSearchTransport().lookup(itunes_id)  # type: ignore[arg-type]


class DefaultCatalogSourceTest(unittest.TestCase):
    def test_default_source_is_credential_free_itunes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(default_catalog_search_source(), iTunesSearchAdapter)
        with patch.dict(os.environ, {"MUSIC_AGENT_CATALOG_PROVIDER": "itunes"}):
            self.assertIsInstance(default_catalog_search_source(), iTunesSearchAdapter)

    def test_music_kit_stays_selectable(self) -> None:
        with patch.dict(os.environ, {"MUSIC_AGENT_CATALOG_PROVIDER": "music_kit"}):
            self.assertIsInstance(default_catalog_search_source(), AppleMusicCatalogAdapter)


if __name__ == "__main__":
    unittest.main()
