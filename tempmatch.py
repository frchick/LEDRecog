import cv2
import numpy as np


#------------------------------------------------------------------------------
# テンプレートマッチングの閾値を超えるピクセルが連続する際に、そのピークを検出する
class MatchInfo:
  def __init__(self, mv, x, y):
    self.mv = mv
    self.x = x
    self.y = y
    self.w = 1
    self.h = 1
    self.label = ''

# テンプレートマッチングの評価値画像から、ピーク検出されたマッチング座標の配列を返す
def make_match_pos(res, threshold):
  w = res.shape[1]
  h = res.shape[0]
  idx_buf = [[-1] * w] * h
  index = 0
  match_infos = []
  for y in range(h):
    for x in range(w):
      mv = res[y][x]
      if (idx_buf[y][x] < 0) and (threshold <= mv):
        match_infos.append(MatchInfo(mv, x, y))
        walk_matchs(res, threshold, idx_buf, x, y, index, match_infos[index])
        index += 1
  return match_infos

# 閾値を超えるピクセルが隣接する際の、ピーク検出
# make_match_pos() のサブルーチン。隣接ピクセルへの再帰処理。
def walk_matchs(res, threshold, idx_buf, x, y, index, match_info):
  # すでにマッチインデックスが割り当てられていたら何もしない
  if 0 <= idx_buf[y][x]:
    return
  # 閾値未満なら何もしない
  mv = res[y][x]
  if mv < threshold:
    return
  # マッチインデックスを共有して、このピクセルがピークなのであればマッチ情報を更新
  idx_buf[y][x] = index
  if match_info.mv < mv:
    match_info.mv = mv
    match_info.x = x
    match_info.y = y
  # 隣接するピクセルに再帰(斜め方向もチェック)
  w = res.shape[1]
  h = res.shape[0]
  if 0 < x:
    walk_matchs(res, threshold, idx_buf, x-1, y, index, match_info)
  if x < (w-1):
    walk_matchs(res, threshold, idx_buf, x+1, y, index, match_info)
  if 0 < y:
    walk_matchs(res, threshold, idx_buf, x, y-1, index, match_info)
  if y < (h-1):
    walk_matchs(res, threshold, idx_buf, x, y+1, index, match_info)
  if (0 < x) and (0 < y):
    walk_matchs(res, threshold, idx_buf, x-1, y-1, index, match_info)
  if (0 < x) and (y < (h-1)):
    walk_matchs(res, threshold, idx_buf, x-1, y+1, index, match_info)
  if (x < (w-1)) and (0 < y):
    walk_matchs(res, threshold, idx_buf, x+1, y-1, index, match_info)
  if (x < (w-1)) and (y < (h-1)):
    walk_matchs(res, threshold, idx_buf, x+1, y+1, index, match_info)

#------------------------------------------------------------------------------
# 検出するテンプレート画像
class TempImage:
  def __init__(self, file_names, label, threshold):
    self.file_names = file_names
    self.label = label
    self.threshold = threshold

tempImages = [
  TempImage([ 'tempmatch_off0.png', 'tempmatch_off1.png' ], 'off', 0.80),
#  TempImage([ 'tempmatch_on_red0.png', 'tempmatch_on_red1.png', 'tempmatch_on_green0.png', 'tempmatch_on_green1.png' ], 'on', 0.70),
]

#------------------------------------------------------------------------------
# 画像からテンプレートパターンとマッチする座標を列挙
def fing_pattern_match_pos(img):
  # グレースケール化して処理
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_w = img_gray.shape[1]
  img_h = img_gray.shape[0]

  # 各テンプレート画像を順に処理する
  match_infos_res = []
  for ti in tempImages:
    # テンプレート画像の高さ、幅を取得する
    temp_img = cv2.imread('./data/samples/' + ti.file_names[0])
    temp_w = temp_img.shape[1]
    temp_h = temp_img.shape[0]
    # ピクセル毎のマッチング度合いを格納するバッファを作成
    res = np.zeros(((img_h - temp_h + 1), (img_w - temp_w + 1), 1), dtype=np.float32)
    # 複数のテンプレート画像を一括して評価
    for file_name in ti.file_names:
      # テンプレート画像の読み込み
      temp_img = cv2.imread('./data/samples/' + file_name)
      temp_img_gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
      # 画像からテンプレート画像とのマッチング度合いを評価
      r = cv2.matchTemplate(img_gray, temp_img_gray, cv2.TM_CCOEFF_NORMED)
      # 複数のテンプレート画像のなかで、最もマッチング度合いが高い値を記録
      res = cv2.max(res, r)

    # マッチング度合いの高い部分を検出する
    # 閾値以上のピクセルが連続する場合には、ピークを取るピクセルにまとめる
    match_infos = make_match_pos(res, ti.threshold)
    for mi in match_infos:
      mi.w = temp_w
      mi.h = temp_h
      mi.label = ti.label
    match_infos_res += match_infos

  return match_infos_res

#------------------------------------------------------------------------------
def main():
  # 画像の読み込み
  img = cv2.imread('./data/samples/tempmatch_cp0.png')
  # テンプレートパターンと一致する位置を取得
  match_infos = fing_pattern_match_pos(img)

  # 検出した部分に赤枠をつける
  for mi in match_infos:
      mv = '{:.2f}'.format(mi.mv)
      print(str(mi.x) + ',' + str(mi.y) + ':' + str(mv) + ':' + mi.label)
      cv2.rectangle(img, (mi.x, mi.y), (mi.x + mi.w, mi.y + mi.h), (0, 0, 255), 1)
      cv2.putText(img, str(mv), (mi.x + mi.w + 1, mi.y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
      cv2.putText(img, mi.label, (mi.x + mi.w + 1, mi.y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

  # 画像の保存
  cv2.imwrite('./tmpmatch_res.png', img)

  # 画像の表示
  cv2.namedWindow('TempMatch', cv2.WINDOW_NORMAL)
  cv2.imshow('TempMatch', img)

  # キー入力待ち(ここで画像が表示される)
  cv2.waitKey()

# エントリーポイント
if __name__ == "__main__":
  main()