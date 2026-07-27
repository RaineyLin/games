# Rainey’s Games

GitHub Pages 瀏覽器遊戲合集。

- **首頁**：https://raineylin.github.io/games/
- **倉庫**：https://github.com/RaineyLin/games

## 遊戲

| 遊戲 | 路徑 | 說明 |
|------|------|------|
| 暗棋 Dark Chess | [`Dark_chess/`](Dark_chess/) | 中國半棋盤，雙人 / 對電腦 |
| 極速飛躍 Motocross | [`Motocross/`](Motocross/) | 山丘越野摩托，30 秒計時競速 |

## 本機預覽

```bash
# 在倉庫根目錄
python3 -m http.server 8080
# 開啟 http://localhost:8080
# 暗棋：http://localhost:8080/Dark_chess/
# Motocross：http://localhost:8080/Motocross/
```

## 更新 Motocross

從 Godot 專案匯出 Web 後覆蓋 `Motocross/`：

```bash
godot --headless --path /path/to/Motocross --export-release "Web" "build/web/index.html"
rsync -a --delete --exclude='*.import' build/web/ /path/to/games/Motocross/
```

## 結構

```
games/
├── index.html
├── README.md
├── .nojekyll
├── Dark_chess/
└── Motocross/
```
