#!/usr/bin/env python3
"""Test FastAPI import and basic functionality"""

try:
    import fastapi
    print(f"FastAPI version: {fastapi.__version__}")

    from fastapi import APIRouter
    print("APIRouter imported successfully")

    # Try to create a router
    router = APIRouter()
    print("APIRouter created successfully")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()