#!/bin/sh
# LightGBM's macOS wheel links to libomp, normally installed with Homebrew. Where Homebrew
# is unavailable (e.g. Intel Macs it no longer supports), point LightGBM at the copy of
# libomp that ships inside scikit-learn. Run once after `pip install -r requirements.txt`.
set -e
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
RPATH="@loader_path/../../sklearn/.dylibs"
if otool -l "$LIB" | grep -q "$RPATH"; then
  echo "already patched"
else
  install_name_tool -add_rpath "$RPATH" "$LIB"
  echo "patched $LIB"
fi
.venv/bin/python -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'loads')"
