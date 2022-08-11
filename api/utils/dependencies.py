from fastapi import Header, HTTPException
from typing import Optional
from config import settings



async def check_api_key(x_api_key: Optional[str] = Header(None)):
    # Check if key "x-api-key" is in headers
    # Dashes in the actual header require respective underscore variable 
    # names in Fastapi: https://learnings.desipenguin.com/post/headers-fastapi/


    print("USED API KEY: ", x_api_key)
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    
    # Check if key is in the list of allowed keys
    for key in settings.API_KEYS:
        if key == x_api_key:
            return
    raise HTTPException(status_code=401, detail="API Key is not correct.")
