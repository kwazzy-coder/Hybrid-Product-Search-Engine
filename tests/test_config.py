import tomllib
from pathlib import PureWindowsPath

from crawler.config import load_config


def test_windows_paths_in_config(tmp_path):
    win = PureWindowsPath(r"C:\Users\Samrudh\AppData\Local\Temp\run")
    text = f'''[crawler]
db_path = "{win.as_posix()}/crawler.db"
cache_dir = 'C:\\Users\\Samrudh\\cache'
'''
    parsed = tomllib.loads(text)
    assert parsed["crawler"]["db_path"] == "C:/Users/Samrudh/AppData/Local/Temp/run/crawler.db"
    assert parsed["crawler"]["cache_dir"] == "C:\\Users\\Samrudh\\cache"


def test_default_sites_load():
    cfg = load_config()
    assert [s.id for s in cfg.sites if s.enabled] == ["libas", "suta", "powerlook", "thebearhouse", "bonkerscorner"]
