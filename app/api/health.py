from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "code": 200,
        "message": "success",
        "data": {
            "status": "UP"
        }
    }