from pydantic import BaseModel

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import settings

from core.runner import faq_inference_runner

router = APIRouter(
    prefix=f"{settings.BASE_URL}",
    tags=["FAQ Inference"],
)


class Input(BaseModel):
    text: str


@router.post("/faq-inference")
async def faq_inference(input: Input):
    """Infers an Input Question from the generated FAQ Dataset.
    
    Args:
        Input: string.

    Returns:
        Output: json dict.
    """

    input_txt = input.text

    response_dict = {}
    response_dict[input_txt] = faq_inference_runner(input_txt)

    return JSONResponse(content=response_dict)
