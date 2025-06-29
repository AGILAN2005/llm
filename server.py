
from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid, json, os, httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# JWT Config
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# File DB
DB_FILE = "users.json"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
app = FastAPI()

# --- Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str
    api_key: Optional[str] = None
    usage: Optional[dict] = {}
    log: Optional[list] = []
    credits: Optional[int] = 1000

class PromptRequest(BaseModel):
    prompt: str

# --- Utility ---
def load_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump({}, f)
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_user(db, username: str):
    if username in db:
        return UserInDB(**db[username])

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db, username: str, password: str):
    user_dict = db.get(username)
    if not user_dict:
        return False
    user = UserInDB(**user_dict)
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- Auth Dependencies ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    db = load_db()
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Routes ---
@app.post("/register")
async def register_user(form: OAuth2PasswordRequestForm = Depends()):
    db = load_db()
    if form.username in db:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed = get_password_hash(form.password)
    api_key = str(uuid.uuid4())
    db[form.username] = {
        "username": form.username,
        "email": None,
        "full_name": form.username,
        "disabled": False,
        "hashed_password": hashed,
        "api_key": api_key,
        "usage": {"count": 0, "last_reset": datetime.utcnow().isoformat()},
        "log": [],
        "credits": 1000
    }
    save_db(db)
    return {"msg": "User registered", "api_key": api_key}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = load_db()
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username},
                                       expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/generate")
async def generate_response(request: PromptRequest, x_api_key: str = Header(...),current_user: User = Depends(get_current_active_user)):
    db = load_db()
    user_found = None
    for user in db.values():
        if user.get("api_key") == x_api_key:
            user_found = user
            break
    if not user_found:
        raise HTTPException(status_code=403, detail="Invalid API key")

    now = datetime.utcnow()
    usage = user_found.get("usage", {})
    last_reset = datetime.fromisoformat(usage.get("last_reset", now.isoformat()))
    count = usage.get("count", 0)
    if (now - last_reset) > timedelta(hours=1):
        count = 0
        last_reset = now
    if count >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if user_found.get("credits", 0) <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post("http://localhost:11434/api/generate", json={
                "model": "llama3.2",
                "prompt": request.prompt,
                "stream": False
            })
        result = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Log + Update usage + Credits
    count += 1
    user_found["usage"] = {"count": count, "last_reset": last_reset.isoformat()}
    user_found["credits"] = user_found.get("credits", 1000) - 1
    log = user_found.get("log", [])
    log.append({"timestamp": now.isoformat(), "prompt": request.prompt})
    user_found["log"] = log[-100:]
    save_db(db)

    return {"response": result.get("response", ""), "remaining_credits": user_found["credits"]}

@app.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/regenerate-api-key")
async def regenerate_api_key(current_user: UserInDB = Depends(get_current_active_user)):
    db = load_db()
    user = db.get(current_user.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_key = str(uuid.uuid4())
    user["api_key"] = new_key
    save_db(db)
    return {"new_api_key": new_key}
