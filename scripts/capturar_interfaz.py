"""capturar_interfaz.py — Captura screenshots reales de las 3 pestañas
principales de la demo (Cámara en vivo, Subir video, Subir imagen) usando
Playwright headless contra el servidor Gradio real corriendo en
localhost:7860, para usar como evidencia gráfica de la interfaz en la tesis.
"""
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "data" / "sustentacion_figs"
OUT.mkdir(exist_ok=True)

TABS = [
    ("📷 Cámara en vivo", "interfaz_camara.png"),
    ("🎬 Subir video", "interfaz_video.png"),
    ("🖼️ Subir imagen", "interfaz_imagen.png"),
]


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto("http://localhost:7860", wait_until="load", timeout=30000)
        time.sleep(3.0)

        for tab_label, fname in TABS:
            # Gradio TabItem se renderiza como un botón con ese texto
            btn = page.get_by_role("tab", name=tab_label)
            if btn.count() == 0:
                # fallback: buscar por texto parcial
                btn = page.locator(f"button:has-text('{tab_label.split(' ', 1)[1]}')")
            btn.first.click()
            time.sleep(1.2)
            page.screenshot(path=str(OUT / fname), full_page=True)
            print(f"✅ {fname}")

        browser.close()


if __name__ == "__main__":
    main()
