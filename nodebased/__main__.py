from . import settingshome
settingshome.apply()
from .app import main
raise SystemExit(main())
