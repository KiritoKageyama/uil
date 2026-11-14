from pathlib import Path
root = Path(r"c:\Users\arinb\Downloads\figpso_uil")
# Import HTML from a separate utf8 file we will write next
html_path = root / "_recovery" / "desktop_index.html"
css_path = root / "_recovery" / "desktop_app.css"
assert html_path.exists(), html_path
assert css_path.exists(), css_path
for src, dst in [
    (html_path, root / "webui" / "desktop" / "index.html"),
    (css_path, root / "webui" / "desktop" / "app.css"),
]:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.write_bytes(src.read_bytes())
    print("installed", dst, dst.stat().st_size)
