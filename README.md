# Rainey’s Games

GitHub Pages 瀏覽器遊戲合集。

- **首頁**：https://raineylin.github.io/games/
- **倉庫**：https://github.com/RaineyLin/games

## 遊戲

| 遊戲 | 路徑 | 說明 |
|------|------|------|
| 暗棋 Dark Chess | [`Dark_chess/`](Dark_chess/) | 中國半棋盤，雙人 / 對電腦 |

## 本機預覽

```bash
# 在倉庫根目錄
python3 -m http.server 8080
# 開啟 http://localhost:8080
# 暗棋：http://localhost:8080/Dark_chess/
```

## 更新暗棋

從 Godot 專案匯出 Web 後覆蓋 `Dark_chess/`：

```bash
godot --headless --path /path/to/Dark_chess --export-release "Web" "build/web/index.html"
rsync -a --delete --exclude='*.import' build/web/ /path/to/games/Dark_chess/
```

## 結構

```
games/
├── index.html          # 遊戲清單首頁
├── README.md
├── .nojekyll
└── Dark_chess/         # Godot HTML5 匯出
    ├── index.html
    ├── index.js
    ├── index.wasm
    ├── index.pck
    └── …
```
