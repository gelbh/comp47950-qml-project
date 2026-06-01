"""FAQ page — static Q&A sourced from ``content.FAQ``."""

from __future__ import annotations

import streamlit as st

from content import FAQ, PAGE_BLURBS, PAGE_TITLES  # type: ignore[import-not-found]

st.title(PAGE_TITLES["faq"])
_blurb = PAGE_BLURBS["faq"].strip()
if _blurb:
    st.markdown(_blurb)

st.sidebar.header("Filter")
query = st.sidebar.text_input("Search", value="").strip().lower()

visible = [
    (q, a)
    for q, a in FAQ
    if not query or query in q.lower() or query in a.lower()
]

if not visible:
    st.info("No questions match the current filter.")
else:
    for idx, (question, answer) in enumerate(visible):
        with st.expander(question, expanded=(idx == 0 and not query)):
            st.markdown(answer)
