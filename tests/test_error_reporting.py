import ast
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, StreamingResponse

from utils.error_handler import ErrorReportingMiddleware


class ErrorReportingTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.add_middleware(ErrorReportingMiddleware)
        # Load the real 500 handler without initializing the project's DB/LLM clients.
        source = ast.parse(Path('main.py').read_text(encoding='utf-8'))
        handler = next(node for node in source.body
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == 'global_exception_handler')
        exec(compile(ast.Module(body=[handler], type_ignores=[]), 'main.py', 'exec'),
             {'app': app, 'Request': Request, 'Response': Response, 'JSONResponse': JSONResponse})

        @app.get('/ok')
        def ok():
            return {'ok': True}

        @app.get('/http')
        def http_error():
            raise HTTPException(401, 'token missing', headers={'X-Test': 'kept'})

        @app.get('/validation')
        def validation(value: int):
            return value

        @app.get('/crash')
        def crash():
            raise ValueError('inference failed')

        @app.get('/stream')
        def stream():
            async def chunks():
                yield 'first\n'
                raise RuntimeError('stream failed')
            return StreamingResponse(chunks())

        self.client = TestClient(app, raise_server_exceptions=False)
        self.report = AsyncMock()
        patcher = patch('utils.error_handler.send_error', self.report)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_is_not_reported(self):
        self.assertEqual(self.client.get('/ok').json(), {'ok': True})
        self.report.assert_not_awaited()

    def test_http_error_preserves_response_and_reports_once(self):
        response = self.client.get('/http')
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {'detail': 'token missing'})
        self.assertEqual(response.headers['X-Test'], 'kept')
        self.report.assert_awaited_once()
        self.assertIn('token missing', self.report.call_args.args[0]['message'])

    def test_validation_and_missing_route_are_reported(self):
        for path, status in [('/validation?value=bad', 422), ('/missing', 404)]:
            with self.subTest(path=path):
                self.report.reset_mock()
                self.assertEqual(self.client.get(path).status_code, status)
                self.report.assert_awaited_once()
                self.assertEqual(self.report.call_args.args[0]['status'], status)

    def test_unhandled_error_is_reported_once(self):
        response = self.client.get('/crash')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {'detail': 'Internal Server Error'})
        self.report.assert_awaited_once()
        payload = self.report.call_args.args[0]
        self.assertEqual(payload['error'], 'ValueError')
        self.assertEqual(payload['message'], 'inference failed')
        self.assertEqual(payload['endpoint'], '/crash')

    def test_error_after_stream_starts_is_reported_once(self):
        self.client.get('/stream')
        self.report.assert_awaited_once()
        self.assertIn('stream failed', self.report.call_args.args[0]['message'])
