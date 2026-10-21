
import sys, os
if __name__ == 'streamlit':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if os.path.abspath(p) != this_dir and p != '']
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    import streamlit as _real_st
    sys.modules['streamlit'] = _real_st
    sys.path = saved_path
    globals().update(_real_st.__dict__)
else:
    import runpy
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    APP_PATH = os.path.join(ROOT_DIR, 'app', 'app.py')
    runpy.run_path(APP_PATH, run_name='__main__')
