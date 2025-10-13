import datetime as dt

import psycopg
from redis.exceptions import RedisError
from fastapi import APIRouter, Response, status

from rag.api.dependencies import get_postgres_conn, get_redis_client
from rag.api.models import HealthResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
def health_check(response: Response):
    services = {
        "redis": get_redis_status(),
        "postgres": get_postgres_status()
    }
    all_healthy = all(v == "healthy" for v in services.values())
    overall_status = "healthy" if all_healthy else "unhealthy"
    response.status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": overall_status,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "services": services,
    }

def get_postgres_status():
    try:
        with get_postgres_conn() as conn:
            conn.execute("SELECT 1")
        return "healthy"
    except psycopg.Error as e:
        return f"{e.__class__.__name__}: {e}"

def get_redis_status():
    try:
        client = get_redis_client()
        client.ping()
        return "healthy"
    except RedisError as e:
        return f"{e.__class__.__name__}: {e}"