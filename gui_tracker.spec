# -*- mode: python ; coding: utf-8 -*-


import mediapipe
import os

# Get the path to the mediapipe module
mediapipe_path = os.path.dirname(mediapipe.__file__)
mediapipe_model_path = os.path.join(mediapipe_path, 'modules')

a = Analysis(
    ['gui_tracker.py'],
    pathex=[],
    binaries=[],
    datas=[(mediapipe_model_path, 'mediapipe/modules')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='gui_tracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
