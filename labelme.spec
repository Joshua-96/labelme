# -*- mode: python -*-
# vim: ft=python

import sys


sys.setrecursionlimit(5000)  # required on Windows


a = Analysis(
    ['labelme/__main__.py'],
    pathex=['labelme'],
    binaries=[],
    datas=[
        ('labelme/config/default_config.yaml', 'labelme/config'),
        ('labelme/icons/*', 'labelme/icons'),
        ('C:/Program Files (x86)/Intel/openvino_2022.1.0.643/runtime/bin/intel64/Release/plugins.xml',
        '.')
    ],
    hiddenimports=['yaml','termcolor','colorama','openvino','pkgutil','OpenGL_accelerate'],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tflite-runtime","PyQt6","cupy"],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='labelme',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon='labelme/icons/icon.ico',
)
app = BUNDLE(
    exe,
    name='Labelme.app',
    icon='labelme/icons/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
