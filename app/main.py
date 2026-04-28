"""
main.py — Entry point for the HERC Research mentions app.
Run directly with:  python main.py
Or as a frozen exe built by PyInstaller.
"""

import sys
import os

# When frozen by PyInstaller, ensure bundled resources are found
if getattr(sys, "frozen", False):
    base_dir = sys._MEIPASS
    sys.path.insert(0, base_dir)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

from gui import App

if __name__ == "__main__":
    app = App()
    app.mainloop()