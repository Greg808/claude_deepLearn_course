# Frontend-Änderungen: Umschalter Hell/Dunkel

Nur das Frontend ist betroffen. Das Backend bleibt unverändert.

## Verhalten
- Ein runder Icon-Button oben rechts (`#themeToggle`) schaltet zwischen dunklem (Standard) und hellem Theme um.
- Das Theme steht als `data-theme="dark|light"` auf `<html>`. Alle Farben kommen aus CSS-Variablen. `:root[data-theme="light"]` überschreibt nur deren Werte.
- Die Wahl wird in `localStorage` (`theme`) gespeichert. Ein kleines Inline-Script im `<head>` setzt sie vor dem ersten Rendern, damit das falsche Theme nicht kurz aufblitzt.
- Übergang: Während des Umschaltens setzt `script.js` für 300 ms die Klasse `theme-transition` auf `<html>`. Dadurch blenden Hintergrund-, Text- und Rahmenfarben weich über. Mond und Sonne drehen und blenden ineinander. Bei `prefers-reduced-motion` gibt es keine Animation.
- Barrierefreiheit: Der Button ist ein echtes `<button>`, also per Tab erreichbar und mit Enter oder Leertaste auslösbar. Er hat einen sichtbaren Fokusring. `aria-label` und `title` nennen die Aktion („Switch to light theme“ bzw. „Switch to dark theme“). Die SVGs sind `aria-hidden`.

## Dateien
- `frontend/index.html`: Button mit Mond- und Sonnen-SVG, Inline-Script im `<head>`, Cache-Version `?v=11`.
- `frontend/style.css`:
  - Variablen für das helle Theme (Slate-Töne, Text `#0f172a` / `#475569` auf `#f1f5f9` / `#ffffff`, Kontrast nach WCAG AA). Die Primärfarbe `#2563eb` bleibt in beiden Themes gleich.
  - Neue Variablen `--code-bg`, `--error-text` und `--success-text` statt fest eingetragener Farben. So bleiben Fehlermeldungen und Code-Blöcke auch auf hellem Hintergrund gut lesbar.
  - `color-scheme` pro Theme, damit Scrollbars und Formularelemente passen.
  - Styles für `.theme-toggle` und die Icon-Animation.
  - Fehler behoben: `blockquote` nutzte die nicht definierte Variable `--primary`, jetzt `--primary-color`.
- `frontend/script.js`: `setupThemeToggle()` schaltet um, speichert die Wahl, aktualisiert das Label und steuert den Übergang.
- `backend/tests/e2e/test_frontend.py`: neuer Test `test_theme_toggle_switches_and_persists` für Klick, Variablenwert, Speichern über einen Reload und Bedienung per Tastatur.
