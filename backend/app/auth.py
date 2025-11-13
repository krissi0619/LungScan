# In your auth.py - Update these functions

def create_user(db: Session, email: str, password: str, full_name: str) -> User:
    """Create new user in database"""
    try:
        # Check if user already exists
        existing_user = get_user_by_email(db, email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user with proper field names
        hashed_password = get_password_hash(password)
        db_user = User(
            email=email,
            hashed_password=hashed_password,  # Make sure this matches your User model
            full_name=full_name,
            created_at=datetime.utcnow()
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

def authenticate_user(db: Session, email: str, password: str):
    """Authenticate user with email and password"""
    try:
        user = get_user_by_email(db, email)
        if not user:
            return False
        
        # Make sure you're using the correct password field name
        if not verify_password(password, user.hashed_password):
            return False
            
        return user
    except Exception as e:
        print(f"Authentication error: {e}")
        return False