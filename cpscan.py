import os
import cv2
import numpy as np
import math

# カンマ逆補正LUT
gamma22LUT = np.array([pow(x/255.0 , 2.2) * 255 for x in range(256)], dtype='uint8')

#------------------------------------------------------------------------------
# 画像を読み込むディレクトリ
images_path = 'data/kazono20230625'

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

# 7セグの座標
class segDef:
  def __init__(self, xx, yy, ww, hh):
    self.x = xx
    self.y = yy
    self.w = ww
    self.h = hh

segs = [
  segDef(684, 477, 20, 30),
  segDef(707, 477, 20, 30),
  segDef(729, 477, 20, 30),
  segDef(751, 477, 20, 30),
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
  for subdir, dirs, files in os.walk(images_path):
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
    lamp_img = img[lp.y: lp.y+lp.h, lp.x: lp.x+lp.w]
    s = read_lamp_color(lamp_img)
    print(', ' + s, end='')
    f.write(', ' + s)
  
  # 7セグの文字を読み取り
  text = ''
  for sg in segs:
    seg_img = img[sg.y: sg.y+sg.h, sg.x: sg.x+sg.w]
    text = text + read_7seg_char(seg_img)
  print(', ' + text)
  f.write(', ' + text + '\n')


#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# 認識文字のセグメント発光パターン(1:発光 / 0:消灯)
#  +-0-+
#  1   2
#  +-3-+
#  4   5
#  +-6-+
class char_pattern:
  def __init__(self, ch, pattern):
    self.ch = ch
    self.pattern = pattern

chars = [
  #                  0, 1,2, 3, 4,5, 6
  char_pattern('0', [1, 1,1, 0, 1,1, 1]),
  char_pattern('1', [0, 0,1, 0, 0,1, 0]),
  char_pattern('2', [1, 0,1, 1, 1,0, 1]),
  char_pattern('3', [1, 0,1, 1, 0,1, 1]),
  char_pattern('4', [0, 1,1, 1, 0,1, 0]),
  char_pattern('5', [1, 1,0, 1, 0,1, 1]),
  char_pattern('6', [0, 1,0, 1, 1,1, 1]),
  char_pattern('6', [1, 1,0, 1, 1,1, 1]),
  char_pattern('7', [1, 0,1, 0, 0,1, 0]),
  char_pattern('7', [1, 1,1, 0, 0,1, 0]),
  char_pattern('8', [1, 1,1, 1, 1,1, 1]),
  char_pattern('9', [1, 1,1, 1, 0,1, 0]),
  char_pattern('9', [1, 1,1, 1, 0,1, 1]),

  char_pattern('A', [1, 1,1, 1, 1,1, 0]),
  char_pattern('E', [1, 1,0, 1, 1,0, 1]),
  char_pattern('F', [1, 1,0, 1, 1,0, 0]),
  char_pattern('H', [0, 1,1, 1, 1,1, 0]),
  char_pattern('L', [0, 1,0, 0, 1,0, 1]),
  char_pattern('U', [0, 1,1, 0, 1,1, 1]),

  char_pattern('-', [0, 0,0, 1, 0,0, 0]),
  char_pattern(' ', [0, 0,0, 0, 0,0, 0]),
]

# 7セグマスク画像
img_7seg_mask = []

# 7セグマスク画像を読み込む
def load_7seg_masks():
  # すでに読み込み済みなら何もしない
  if 0 == len(img_7seg_mask):
    # 7セグマスク画像(グレースケールで読み込む)
    for i in range(7):
      img = cv2.imread('data/res/seg' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
      img = cv2.LUT(img, gamma22LUT)
      img_7seg_mask.append(img)

# 7セグ文字の読み取り
def read_7seg_char(img):
  # 7セグマスク画像を読み込む
  load_7seg_masks()
  # 輝度をリニア・グレースケール化
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.LUT(img, gamma22LUT)
  # コントラスト強める
  # ヒストグラムのピークをブラックレベルとする
  hist = cv2.calcHist([img], [0], None, [256], [0, 256])
  black_floor = 0
  hist_max = 0
  for i in range(256):
    if hist_max < hist[i]:
      hist_max = hist[i]
      black_floor = i
  alpha = 2.0 * 255 / (255 - black_floor)
  img = cv2.convertScaleAbs(img, alpha=alpha, beta=-alpha*black_floor)
  # セグメント毎にマスクされた輝度と、その反転輝度を計算
  v = [ 0.0 ] * 7
  iv = [ 0.0 ] * 7
  inv_img = cv2.bitwise_not(img)
  for i in range(7):
    masked_img = cv2.multiply(img, img_7seg_mask[i], scale=1.0/255.0)
    v[i] = np.sum(masked_img)
    masked_img = cv2.multiply(inv_img, img_7seg_mask[i], scale=1.0/255.0)
    iv[i] = np.sum(masked_img)

  # どの文字かを判定する
  max_vv = 0.0
  ch = '?'
  for c in chars:
    vv = 0
    for i in range(7):
      if 0 < c.pattern[i]:
        vv += v[i]
      else:
        vv += iv[i]
    if max_vv < vv:
      max_vv = vv
      ch = c.ch
  return ch

# エントリーポイント
if __name__ == "__main__":
  main()
