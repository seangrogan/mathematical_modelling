#!D:\python\mathematical_modelling\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'PuLP==1.6.10','console_scripts','pulptest'
__requires__ = 'PuLP==1.6.10'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('PuLP==1.6.10', 'console_scripts', 'pulptest')()
    )
