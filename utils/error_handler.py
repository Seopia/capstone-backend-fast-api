import httpx


import httpx
import traceback
import os


class ErrorReportingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        status = 200
        reported = False

        async def report(error, message, error_status, file=None, line=None, stack=None):
            nonlocal reported
            reported = True

            await send_error({
                "service": "emotion-chatbot",
                "status": error_status,
                "error": error,
                "message": message,
                "endpoint": scope["path"],
                "file": file,
                "line": line,
                "traceback": stack
            })

        async def send_response(message):
            nonlocal status

            if message["type"] == "http.response.start":
                status = message["status"]

            elif message["type"] == "http.response.body" and status >= 400 and not reported:
                await report(
                    "HTTPError",
                    message.get("body", b"").decode("utf-8", errors="replace"),
                    status
                )

            await send(message)

        try:
            await self.app(scope, receive, send_response)

        except Exception as exc:
            if not reported:
                tb = traceback.extract_tb(exc.__traceback__)

                project_root = os.getcwd()
                frame = None

                for item in reversed(tb):
                    file_path = os.path.abspath(item.filename)

                    if os.path.commonpath([project_root, file_path]) == project_root:
                        frame = item
                        break

                file = None
                line = None

                if frame:
                    file = os.path.relpath(
                        frame.filename, project_root
                    ).replace("\\", "/")
                    line = frame.lineno

                await report(
                    type(exc).__name__,
                    str(exc),
                    500,
                    file,
                    line,
                    traceback.format_exception_only(type(exc), exc)[-1].strip()
                )

            raise


async def send_error(error):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post("http://localhost:5678/webhook/fastapi-error", json=error)
            response.raise_for_status()
    except Exception as e:
        print("Error 전달 중 Error 발생....", e)
