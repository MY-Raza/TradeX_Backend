from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.model_schema import (
    AllModelsResponse,
    ModelResultDetail,
    ModelTypeOptions,
    PaginatedModelResults,
)
from app.services import model_service

router = APIRouter(prefix="/models", tags=["Models"])

# Reusable type alias for the DB dependency
DB = Annotated[AsyncSession, Depends(get_db)]


# ===========================================================================
# GET /models   ← returns ALL models from both ml_results and dl_results
# ===========================================================================

@router.get(
    "",
    response_model=AllModelsResponse,
    summary="Get all models",
    description=(
        "Returns every row from both **ml_results** and **dl_results** tables "
        "in a single response, grouped by type."
    ),
)
async def get_all_models(db: DB) -> AllModelsResponse:
    return await model_service.get_all_models(db)


# ===========================================================================
# GET /models/types   ← MUST come before /{model_type}
# ===========================================================================

@router.get(
    "/types",
    response_model=ModelTypeOptions,
    summary="Get available model types",
    description=(
        "Returns the list of available model type identifiers. "
        "Currently `ml` (machine-learning) and `dl` (deep-learning). "
        "Use these values as the `{model_type}` path segment."
    ),
)
async def list_model_types() -> ModelTypeOptions:
    return await model_service.get_model_type_options()


# ===========================================================================
# GET /models/{model_type}
# ===========================================================================

@router.get(
    "/{model_type}",
    response_model=PaginatedModelResults,
    summary="List model results",
    description=(
        "Returns a paginated list of back-test results for the given model type. "
        "`model_type` must be **ml** or **dl** (case-insensitive). "
        "All query parameters are optional and combinable."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid model_type supplied.",
            "content": {
                "application/json": {
                    "example": {"detail": "model_type must be 'ml' or 'dl'."}
                }
            },
        }
    },
)
async def list_model_results(
    model_type: str,
    db: DB,
    search: Annotated[
        Optional[str],
        Query(description="Partial case-insensitive match on model_name."),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-based)."),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=100, alias="page_size", description="Items per page (max 100)."),
    ] = 20,
) -> PaginatedModelResults:
    if model_type.lower() not in ("ml", "dl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_type must be 'ml' or 'dl'.",
        )
    return await model_service.get_model_results(
        db,
        model_type=model_type,
        search=search,
        page=page,
        page_size=page_size,
    )


# ===========================================================================
# GET /models/{model_type}/{model_name}
# ===========================================================================

@router.get(
    "/{model_type}/{model_name}",
    response_model=ModelResultDetail,
    summary="Get model result detail",
    description=(
        "Returns the full detail record for a single model run. "
        "Includes all financial metrics, risk metrics, and streak data."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid model_type supplied.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "No result found for the given model_name.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model 'random_forest_clf_20260316_120233' not found in ml_results."
                    }
                }
            },
        },
    },
)
async def get_model_result(
    model_type: str,
    model_name: str,
    db: DB,
) -> ModelResultDetail:
    if model_type.lower() not in ("ml", "dl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_type must be 'ml' or 'dl'.",
        )
    result = await model_service.get_model_result_by_name(db, model_type, model_name)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name!r} not found in {model_type.lower()}_results.",
        )
    return result