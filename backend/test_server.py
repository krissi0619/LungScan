from fastapi import FastAPI, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock user database
users_db = []

@app.post("/api/register")
async def register(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    """Simple register endpoint for testing"""
    print(f"Registration attempt: {email}, {full_name}")
    
    # Check if user exists
    if any(user['email'] == email for user in users_db):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = {
        "id": len(users_db) + 1,
        "full_name": full_name,
        "email": email,
        "password": password  # In real app, hash this!
    }
    users_db.append(user)
    
    return {
        "message": "User created successfully",
        "user_id": user["id"],
        "email": user["email"]
    }

@app.post("/api/login")
async def login(
    email: str = Form(...),
    password: str = Form(...)
):
    """Simple login endpoint for testing"""
    print(f"Login attempt: {email}")
    
    user = next((u for u in users_db if u['email'] == email and u['password'] == password), None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    return {
        "access_token": "mock-jwt-token-for-testing",
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"]
        }
    }

@app.get("/api/debug/users")
async def debug_users():
    """Debug endpoint to see registered users"""
    return {
        "total_users": len(users_db),
        "users": users_db
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)