import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
from app import SourceUnit, passages_from_units, split_text


class FakeEmbeddings:
    def embed_content(self, model, contents):
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[1.0, 0.0])
                for _ in contents
            ]
        )


class FakeModels:
    def __init__(self, answer):
        self.answer = answer

    def embed_content(self, model, contents):
        return FakeEmbeddings().embed_content(model, contents)

    def generate_content(self, **kwargs):
        return SimpleNamespace(
            text=self.answer
        )


class FakeClient:
    def __init__(self, answer="[S1] Grounded answer."):
        self.models = FakeModels(answer)


class FakeDatabaseDriver:
    def __init__(self):
        self.statements = []

    def execute(self, query, parameters=()):
        self.statements.append((query, parameters))

    def executemany(self, query, parameters):
        self.statements.append((query, parameters))


class SplitTextTests(unittest.TestCase):
    def test_splits_long_text_with_overlap(self):
        text = "abcdefghij klmnopqrst uvwxyz"
        chunks = split_text(text, chunk_size=12, overlap=3)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][0][-3:], chunks[1][0][:3])
        self.assertEqual(chunks[0][1], 0)
        self.assertEqual(chunks[-1][2], len(text))

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=10, overlap=10)


class CitationTests(unittest.TestCase):
    def test_text_passages_include_line_ranges(self):
        units = [SourceUnit("First line\nSecond line\nThird line", "Document", 8)]

        passages = passages_from_units(units)

        self.assertEqual(passages[0][1], "Lines 8-10")

    def test_heading_passages_keep_section_citation(self):
        units = [SourceUnit("# Overview\nRelevant content.", "Overview", 1)]

        passages = passages_from_units(units)

        self.assertEqual(passages[0][1], 'Section "Overview"')


class IndexingTests(unittest.TestCase):
    def test_index_persists_passages_and_skips_duplicate_content(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "test.db"
            with patch.object(app, "DATABASE_PATH", database):
                app.initialize_db()
                client = FakeClient()
                content = b"First line\nSecond line\nThird line"

                count, added = app.index_document(client, "notes.txt", content)
                duplicate_count, duplicate_added = app.index_document(
                    client, "renamed.txt", content
                )
                retrieved = app.retrieve(client, "a related question")

        self.assertEqual(count, 1)
        self.assertTrue(added)
        self.assertEqual(duplicate_count, 0)
        self.assertFalse(duplicate_added)
        self.assertEqual(retrieved[0].document_name, "notes.txt")
        self.assertEqual(retrieved[0].locator, "Lines 1-3")

    def test_answer_includes_only_valid_source_citations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "test.db"
            with patch.object(app, "DATABASE_PATH", database):
                app.initialize_db()
                app.index_document(
                    FakeClient(), "notes.txt", b"First line\nSecond line"
                )
                with patch.object(app, "rewrite_query", return_value="question"):
                    answer, citations = app.generate_answer(
                        FakeClient("[S1] Grounded answer. [S99]"), "question", []
                    )

        self.assertEqual(answer, "[S1] Grounded answer.")
        self.assertEqual(
            citations,
            [{
                "source_id": "S1",
                "document_name": "notes.txt",
                "locator": "Lines 1-2",
            }],
        )

    def test_existing_passages_are_reembedded_after_model_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "test.db"
            with patch.object(app, "DATABASE_PATH", database):
                app.initialize_db()
                client = FakeClient()
                app.index_document(client, "notes.txt", b"First line")

                with patch.object(app, "EMBEDDING_MODEL", "new-gemini-embedding"):
                    app.retrieve(client, "question")
                    with app.db_connection() as connection:
                        stored_model = connection.execute(
                            "SELECT value FROM app_settings WHERE key = 'embedding_model'"
                        ).fetchone()["value"]

        self.assertEqual(stored_model, "new-gemini-embedding")


class DeploymentTests(unittest.TestCase):
    def test_postgres_adapter_translates_placeholders(self):
        driver = FakeDatabaseDriver()
        connection = app.DatabaseConnection(driver, is_postgres=True)

        connection.execute("SELECT * FROM documents WHERE id = ?", ("id",))

        self.assertEqual(
            driver.statements,
            [("SELECT * FROM documents WHERE id = %s", ("id",))],
        )

    def test_hosted_app_requires_api_key_and_persistent_database(self):
        settings = {"GEMINI_API_KEY": "key", "DATABASE_URL": "postgresql://db"}

        with patch.object(app, "get_setting", side_effect=settings.get):
            self.assertTrue(app.app_is_configured())
        with patch.object(app, "get_setting", return_value=None):
            self.assertFalse(app.app_is_configured())


if __name__ == "__main__":
    unittest.main()
