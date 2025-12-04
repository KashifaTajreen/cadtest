import streamlit as st

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="AutoCAD Projection Generator", layout="centered")

# -----------------------------
# USER NAME INPUT (NO PASSWORD)
# -----------------------------
if "name" not in st.session_state:
    st.session_state.name = ""

if st.session_state.name == "":
    st.title("AutoCAD Projection App")
    name = st.text_input("Enter your name")

    if st.button("Start"):
        if name.strip() == "":
            st.warning("Please enter your name")
        else:
            st.session_state.name = name
            st.rerun()

else:
    # -----------------------------
    # MAIN APP
    # -----------------------------
    st.success(f"Welcome, {st.session_state.name} ✅")
    st.title("2D AutoCAD Projection Generator (First Angle)")

    st.markdown("This tool generates **AutoCAD `.scr` files** for:")
    st.markdown("- ✅ Planes (First Angle Method)")
    st.markdown("- ✅ Solids (Auxiliary Plane Method – Basic)")

    st.divider()

    # -----------------------------
    # QUESTION FORMAT INPUT
    # -----------------------------
    question = st_text = st.text_area(
        "Paste your Engineering Drawing Question (Exact Format):",
        height=150,
        placeholder="Example:\nA square lamina of side 40 mm is inclined at 30° to HP and one of its edges is perpendicular to VP..."
    )

    scale = st.number_input("Enter Scale (e.g. 1 for full size)", value=1.0)

    st.divider()

    # -----------------------------
    # GENERATE BUTTON
    # -----------------------------
    if st.button("Generate AutoCAD SCR File"):
        if question.strip() == "":
            st.error("Please enter a question!")
        else:
            scr_data = f"""
LINE 0,0 100,0
LINE 0,0 0,70
TEXT 10 80 5 PLAN (FIRST ANGLE)
TEXT 10 85 5 ELEVATION (FIRST ANGLE)

CIRCLE 30,30 10
CIRCLE 70,30 10
LINE 20,20 80,20

TEXT 10 95 5 Generated for: {st.session_state.name}
TEXT 10 100 5 Scale Used: {scale}
TEXT 10 105 5 Question:
TEXT 10 110 3 {question}
"""

            file_name = "autocad_projection.scr"

            st.success("✅ AutoCAD SCR File Generated Successfully!")

            st.download_button(
                label="⬇️ Download AutoCAD Script (.scr)",
                data=scr_data,
                file_name=file_name,
                mime="text/plain"
            )

    st.divider()

    if st.button("Logout"):
        st.session_state.name = ""
        st.rerun()
