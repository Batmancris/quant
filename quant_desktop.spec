# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project_root = Path(__file__).resolve().parent

a = Analysis(
    [str(project_root / 'app.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / 'docs'), 'docs'),
        (str(project_root / 'greenpower_demo'), 'greenpower_demo'),
    ],
    hiddenimports=['pandas'],
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
    name='quant_desktop',
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
