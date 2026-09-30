from fastapi import FastAPI

from app.api.health import router as health_router
from app.config.settings import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["Health"],
)


@app.get("/")
async def root():
    return {
        "code": 200,
        "message": "SmartMall Agent is running",
        "data": None,
    }