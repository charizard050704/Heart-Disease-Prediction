import hashlib
import streamlit as st 
import db  # use the updated db.py

# ------------------ Password Hashing ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(email, password):
    """
    Check if email + password match.
    Returns full user row if correct, else None.
    Row format: (email, username, password_hash, role, age, gender)
    """
    user = db.get_user_by_email(email)
    if user and user[2] == hash_password(password):
        return user
    return None

# ------------------ Login / Register Screen ------------------
def login_screen():
    st.markdown("<h2 style='text-align:center;'>🔐 Login / Register</h2>", unsafe_allow_html=True)
    choice = st.radio("Choose Action:", ["Login", "Register"], horizontal=True)

    # For both flows we ask for email + password. For register we also ask for display name.
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Login":
        if st.button("Login"):
            if not email or not password:
                st.error("Please enter both email and password.")
                return
            user = verify_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.email = user[0]      # email
                st.session_state.username = user[1]   # display name
                st.session_state.role = user[3]       # doctor / patient
                st.session_state.age = user[4]
                st.session_state.gender = user[5]

                if st.session_state.role == "doctor":
                    st.success("✅ Logged in successfully as Doctor")
                else:
                    st.success("✅ Logged in successfully as Patient")
            else:
                st.error("❌ Invalid email or password")

    else:  # Register
        username = st.text_input("Display Name")
        role = st.radio("Register as:", ["Patient", "Doctor"], horizontal=True)

        age, gender = None, None
        if role == "Patient":
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.radio("Gender", ["Male", "Female", "Other"], horizontal=True)

        if st.button("Register"):
            if not email or not password or not username:
                st.error("Please provide email, display name and password.")
                return

            success, err = db.add_user(email, username, hash_password(password), role.lower(), age, gender)
            if success:
                st.success(f"✅ {role} registered. You can now login.")
            else:
                st.error(f"⚠️ {err}")
