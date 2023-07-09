import os
import cv2
import numpy
import math

# LEDの座標
class lampDef:
    def __init__(self, name, xp, yp, ww, hh):
        self.name = name
        self.x = xp
        self.y = yp
        self.w = ww
        self.h = hh

lamps = [
  lampDef('電源',    903, 566, 12, 12),
  lampDef('No1運転', 800, 448, 12, 12),
  lampDef('No1停止', 800, 468, 12, 12),
  lampDef('No1故障', 800, 485, 12, 12),
  lampDef('No2運転', 845, 448, 12, 12),
  lampDef('No2停止', 845, 467, 12, 12),
  lampDef('No2故障', 845, 484, 12, 12),
  lampDef('自動',    847, 512, 12, 12),
  lampDef('試験',    847, 530, 12, 12),
  lampDef('交互No1', 886, 512, 12, 12),
  lampDef('交互No1', 886, 530, 12, 12),
]

# main 関数
def main():
  # 出力ファイル
  f = open('resoult.csv', 'w')

  # ヘッダ
  print('画像', end='')
  f.write('画像')
  for lp in lamps:
    print(', ' + lp.name, end='')
    f.write(', ' + lp.name)
  print('')
  f.write('\n')

  # サブディレクトリの画像を列挙して処理
  for subdir, dirs, files in os.walk('data/kazono20230625'):
#  for subdir, dirs, files in os.walk('data/kazono'):
    for file in files:
      if file.lower().endswith('.jpg'):
        file_path = subdir + os.sep + file
        read_control_panel(file_path, f)

# 制御盤の画像から、複数のランプの状態を調べる
def read_control_panel(file_path, f):
  # 画像を読み込む
  print(file_path, end='')
  f.write(file_path)
  img = cv2.imread(file_path)

  # ランプの状態を読み取り
  for lp in lamps:
    lamp_image = img[lp.y: lp.y+lp.h, lp.x: lp.x+lp.w]
    s = read_lamp_color(lamp_image)
    print(', ' + s, end='')
    f.write(', ' + s)
  print('')
  f.write('\n')

# LEDランプの消灯(off), 点灯色(r,g,b)を調べて返す
def read_lamp_color(img):
  # BGRからHSVに変換する
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
  # 色相HをXY座標に変換
  h, w, ch = hsv.shape
  col = [ '' ] * (h*w)                # 無彩色(k,w) or カラー判定(r,g,b)
  colored_pixel_count = [ 0, 0, 0 ]   # R,G,Bのピクセル数のカウント
  i = 0
  for y in range(h):
    for x in range(w):
      px = hsv[y,x]
      Hd = 360 * px[0] / 256          # 色相(DEG)
      S = px[1] / 255                 # 彩度
      V = px[2] / 255                 # 明度

      if V < 0.50:    # 暗いピクセルは黒
        col[i] = 'k'
      elif S < 0.25:  # グレー～白のピクセルは白
        col[i] = 'w'
      else:           # あとはRGB
        if (60 < Hd) and (Hd <= 180):
          col[i] = 'g'
          colored_pixel_count[0] += 1
        elif (180 < Hd) and (Hd <= 300):
          col[i] = 'b'
          colored_pixel_count[1] += 1
        else:
          col[i] = 'r'
          colored_pixel_count[2] += 1
      i += 1

  # 色付きピクセルが極端に少なければ、OFFと判定(とりあえず20%は必要とする)
  s = colored_pixel_count[0] + colored_pixel_count[1] + colored_pixel_count[2]
  if s < (h*w*0.2):
    return 'off'

  # 色付きピクセルから、多い色を返す
  max_count = 0
  max_color = 0
  for i in range(3):
    if max_count <= colored_pixel_count[i]:
      max_count = colored_pixel_count[i]
      max_color = i
  color_code = [ 'g', 'b', 'r' ]
  return color_code[max_color]

# エントリーポイント
if __name__ == "__main__":
  main()
