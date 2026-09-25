# DICOM ビューワ 使い方

## 実行
    pip install -r requirements.txt
    python dicom_viewer.py                # 同階層の DICOM/DICOM を自動読込
    python dicom_viewer.py <DICOMフォルダ>  # フォルダ指定 (Ctrl+O でも可)

## 操作
| 操作 | 内容 |
|---|---|
| ホイール | スライス送り (Shift で5枚ずつ) |
| Ctrl+ホイール | ズーム (カーソル位置基準) |
| 右ドラッグ | ウィンドウ調整 (横=幅W / 縦=レベルL) |
| 中ドラッグ | パン |
| 左クリック/ドラッグ | 選択中ツールの動作 |
| C / W / H / D / E | ツール切替: 位置(十字線) / 窓 / パン / 距離計測 / 楕円ROI |
| 1〜6 | プリセット (DICOM既定, 脳, 硬膜下, 軟部組織, 骨, 肺) |
| I / X / R | 白黒反転 / 十字線表示 / ズーム解除 |
| P | 患者名・ID・生年月日の表示切替 (既定は非表示) |
| T | DICOMタグ一覧 (検索付き、スライス連動) |
| L | Axialのみ / MPR 3面 切替 |
| Del / Ctrl+Del | 現スライスの計測消去 / 全消去 |
| Ctrl+S | 画面をPNG保存 (レポート用) |
