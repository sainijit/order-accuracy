#!/usr/bin/env python3
import httpx
import asyncio
import json

async def test_vlm():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            'http://ovms-vlm:8000/v3/chat/completions',
            json={
                'model': 'Qwen/Qwen2.5-VL-7B-Instruct',
                'messages': [{'role': 'user', 'content': 'Hello, how are you?'}],
                'max_tokens': 10
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_vlm())
