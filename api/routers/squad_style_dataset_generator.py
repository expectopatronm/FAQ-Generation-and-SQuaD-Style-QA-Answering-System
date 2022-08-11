from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from config import settings

from core.qg_qae.utils import preprocess_corpus
from core.runner import squad_style_dataset_generator

router = APIRouter(
    prefix=f"{settings.BASE_URL}",
    tags=["SQuAD Style Dataset Generator"],
)


@router.post("/generate-squad-style-dataset")
async def generate_questions(file: UploadFile = File(...)):
    """Generates and Evaluates Questions from a Context.

    Args:
        Input: .txt file.

    Returns:
        Output: json dict.
    """
    corpus_binary = await file.read()
    corpus_decoded = corpus_binary.decode()
    corpus = preprocess_corpus(corpus_decoded)

    corpus_list = []
    for paragraph in corpus:
        corpus_list.append(squad_style_dataset_generator(paragraph))

    return JSONResponse(content=corpus_list)
