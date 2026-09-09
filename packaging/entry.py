import sys
if len(sys.argv) >= 4 and sys.argv[1] == '--apply-portable-update':
    from nodebased.portable import helper_main
    raise SystemExit(helper_main(sys.argv[2], sys.argv[3], sys.argv[4:]))
from nodebased.app import main
raise SystemExit(main())
